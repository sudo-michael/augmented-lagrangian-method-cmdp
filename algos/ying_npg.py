import tqdm
from algos.base_algo import Algo
import time
from typing import Optional

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv
from flax import struct


class Policy(struct.PyTreeNode):
    num_actions: int
    functional_representation: str = struct.field(pytree_node=False)

    @classmethod
    def create(cls, functional_representation: str = "softmax", num_actions=0):
        return cls(
            functional_representation=functional_representation, num_actions=num_actions
        )

    @jax.jit
    def __call__(self, theta):
        if self.functional_representation == "softmax":
            return jax.nn.softmax(theta, axis=1)
        else:
            return theta

    def sample(self, key, theta, state):
        if self.functional_representation == "softmax":
            pi = jax.nn.softmax(theta, axis=1)
        else:
            pi = theta
        action = jax.random.choice(key, a=self.num_actions, p=pi[state])
        return action


class YINGNPGPD(Algo):
    iterations_outer: int
    iterations_inner: int
    init_lambda: float
    tau: float
    # beta: float
    alpha: float
    functional_representation: str = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        iterations_outer,
        iterations_inner,
        init_lambda,
        tau,
        # beta,
        alpha,
        functional_representation,
    ):
        return cls(
            iterations_outer=iterations_outer,
            iterations_inner=iterations_inner,
            init_lambda=init_lambda,
            tau=tau,
            # beta=beta,
            alpha=alpha,
            functional_representation=functional_representation,
        )

    def npg_subroutine(self, env, pi, lambda_t, tau, T):
        pi_t = pi
        # NOT USING ETA
        eta = (1 - env.gamma) / tau
        power = 1.0 - (eta * tau) / (1.0 - env.gamma)
        # power == 0
        for _ in range(T):
            # prevent log(0)
            pi_safe = jnp.clip(pi_t, 1e-12, 1.0)
            entropy = -tau * jnp.sum(pi_safe * jnp.log(pi_safe), axis=1, keepdims=True)
            Q = env.calc_qpi(pi_t, env.r + lambda_t * env.c + entropy)

            logits = (eta / (1.0 - env.gamma)) * Q
            # exp_part = jax.nn.softmax(logits, axis=1)

            # pi_safe = jnp.clip(pi_t, 1e-12, 1.0)
            # base = pi_t ** power

            # unnorm = base * exp_part
            # pi_t = unnorm / jnp.sum(unnorm, axis=1, keepdims=True)
            pi_tp1 = jax.nn.softmax(logits, axis=1)

            # if jnp.any(jnp.isnan(pi_tp1.flatten())):
            #     print("NaNs found!")
            #     breakpoint()
            pi_t = pi_tp1

        return pi_t

    def project_dual_variable(self, mu, low, high):
        return jnp.minimum(jnp.maximum(mu, 0.0), jnp.maximum(high, 0.0))

    def train(
        self,
        log_dir: str,
        env: TabularEnv,
        env_id: str,
        pi_star: Optional[jnp.ndarray] = None,
        seed: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        key = jax.random.PRNGKey(seed)
        v_r_pis = []
        v_c_pis = []
        lambdas = []
        gradient_evaluations_per_iteration = []

        theta_t = jnp.zeros_like(env.r)
        lambda_tm1 = self.init_lambda
        lambda_t = self.init_lambda

        pi_t = jax.nn.softmax(theta_t, axis=1)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lambda_t)

        start_time = time.time()

        for t in range(self.iterations_outer):
            beta = ((t + 1) - 1) / ((t + 1) + 2)
            mu_t = lambda_t + beta * (lambda_t - lambda_tm1)

            # NPG sub routine
            tilde_pi_t = self.npg_subroutine(
                env, pi_t, mu_t, self.tau, self.iterations_inner
            )

            # compute approximate gradient
            vc = env.rho @ env.calc_vpi(tilde_pi_t, env.c)
            tilde_nabla_D_mu = vc - env.b

            # dual variable update
            mu_max = (2 + 2 * self.tau * jnp.log(env.A)) / (1 - env.gamma)
            lambda_tp1 = self.project_dual_variable(
                mu_t - self.alpha * tilde_nabla_D_mu, low=0.0, high=mu_max
            )

            pi_t = tilde_pi_t
            lambda_tm1 = lambda_t
            lambda_t = lambda_tp1

            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lambda_t.item())
            gradient_evaluations_per_iteration.append(self.iterations_inner)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }

        pi_T = self.npg_subroutine(env, pi_t, mu_t, self.tau, self.iterations_inner)

        self.save_log(
            log_info=log_info,
            log_dir=log_dir,
            env=env,
            env_id=env_id,
            pi_T=pi_T,
            pi_star=pi_star,
            seed=seed,
            elapsed_time=end_time - start_time,
        )

        return True
