from functools import partial

import numpy as np
import tqdm
from algos.base_algo import Algo
import time
from typing import Optional

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv
from flax import struct


class Policy(struct.PyTreeNode):
    functional_representation: str = struct.field(pytree_node=False)

    @classmethod
    def create(cls, functional_representation: str = "softmax"):
        return cls(functional_representation)

    @jax.jit
    def __call__(self, theta):
        if self.functional_representation == "softmax":
            return jax.nn.softmax(theta, axis=1)
        else:
            return theta


class ALMPGDebug(Algo):
    iterations_outer: int
    iterations_inner: int
    init_lambda: float
    init_tau: float
    beta: float
    max_tau: float
    eta: float
    tau_stratergy: int
    inner_grad_norm: float
    functional_representation: str = struct.field(pytree_node=False)
    algo_inner: str = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        iterations_outer,
        iterations_inner,
        init_lambda,
        init_tau,
        beta,
        max_tau,
        eta,
        tau_stratergy,
        functional_representation,
        algo_inner,
        inner_grad_norm,
    ):
        return cls(
            iterations_outer=iterations_outer,
            iterations_inner=iterations_inner,
            init_lambda=init_lambda,
            init_tau=init_tau,
            beta=beta,
            max_tau=max_tau,
            eta=eta,
            tau_stratergy=tau_stratergy,
            functional_representation=functional_representation,
            algo_inner=algo_inner,
            inner_grad_norm=inner_grad_norm,
        )

    def aug_lag(self, theta, lmbda, tau, env, policy):
        pi = policy(theta)
        d_pi = env.calc_dpi(pi)
        mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

        def F(mu_pi):
            vr_pi = jnp.dot(mu_pi.flatten(), env.r.flatten()) / (1 - env.gamma)
            vc_pi = jnp.dot(mu_pi.flatten(), env.c.flatten()) / (1 - env.gamma)

            return vr_pi + (tau / 2) * (
                -(jnp.minimum(0, -lmbda / tau + (vc_pi - env.b)) ** 2)
                + lmbda**2 / tau**2
            )

        return F(mu_pi)

    @jax.jit
    def grad_aug_lag(self, theta, lmbda, tau, env, policy):
        pi = policy(theta)
        d_pi = env.calc_dpi(pi)
        mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

        def F(mu_pi):
            vr_pi = jnp.dot(mu_pi.flatten(), env.r.flatten()) / (1 - env.gamma)
            vc_pi = jnp.dot(mu_pi.flatten(), env.c.flatten()) / (1 - env.gamma)

            return vr_pi + (tau / 2) * (
                -(jnp.minimum(0, -lmbda / tau + (vc_pi - env.b)) ** 2)
                + lmbda**2 / tau**2
            )

        Gamma_pi = jax.grad(F)(mu_pi)

        q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)

        @jax.grad
        def pg(theta):
            return jnp.dot(
                d_pi,
                jnp.einsum("sa, sa->s", q_Gamma_pi, policy(theta)),
            )

        return pg(theta)

    def alm_npg(self, env, policy, theta_t, lambda_t, tau_t):
        @jax.jit
        def q_Gamma_pi(theta, lmbda, tau):
            pi = policy(theta)
            d_pi = env.calc_dpi(pi)
            mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

            def F(mu_pi):
                vr_pi = jnp.dot(mu_pi.flatten(), env.r.flatten()) / (1 - env.gamma)
                vc_pi = jnp.dot(mu_pi.flatten(), env.c.flatten()) / (1 - env.gamma)

                return vr_pi + (tau / 2) * (
                    -(jnp.minimum(0, -lmbda / tau + (vc_pi - env.b)) ** 2)
                    + lmbda**2 / tau**2
                )

            Gamma_pi = jax.grad(F)(mu_pi)

            q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)

            return q_Gamma_pi

        @jax.jit
        def update(theta_inner, lambda_t, tau_t):
            df = partial(q_Gamma_pi, lmbda=lambda_t, tau=tau_t)
            theta_inner += self.eta * df(theta_inner)
            return theta_inner

        theta_k = theta_t
        for _ in range(self.iterations_inner):
            theta_k = update(theta_k, lambda_t, tau_t)

            if (
                jnp.linalg.norm(
                    self.grad_aug_lag(theta_k, lambda_t, tau_t, env, policy)
                )
                < self.inner_grad_norm
            ):
                break

        return theta_k

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
        taus = []
        aug_lag_grad_norms = []
        gradient_evaluations_per_iteration = []

        policy = Policy(self.functional_representation)
        if self.functional_representation == "softmax":
            theta_t = jnp.zeros_like(env.r)
        else:
            theta_t = jnp.ones_like(env.r) / env.A
        lambda_t = self.init_lambda
        tau_t = self.init_tau

        pi_t = policy(theta_t)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        aug_lag_grad_norms.append(
            jnp.linalg.norm(
                self.grad_aug_lag(theta_t, lambda_t, tau_t, env, policy)
            ).item()
        )
        taus.append(tau_t)
        lambdas.append(lambda_t)

        @jax.jit
        def constraint(theta):
            pi = policy(theta)
            vc = env.rho @ env.calc_vpi(pi, env.c)
            return vc - env.b

        start_time = time.time()

        for t in tqdm.tqdm(range(self.iterations_outer), desc="T", leave=False):
            theta_tp1 = self.alm_npg(env, policy, theta_t, lambda_t, tau_t)

            # 2. dual update
            lambda_tp1 = jnp.maximum(lambda_t - tau_t * constraint(theta_tp1), 0)

            # 3. increase tau
            if self.tau_stratergy == 1:
                if jnp.abs(constraint(theta_tp1)) > 0.1 * jnp.abs(constraint(theta_t)):
                    tau_tp1 = min(tau_t * self.beta, self.max_tau)
                else:
                    tau_tp1 = tau_t
            elif self.tau_stratergy == 2:
                tau_tp1 = jnp.clip(
                    self.beta * -min(0, constraint(theta_tp1)),
                    self.init_tau,
                    self.max_tau,
                )
            elif self.tau_stratergy == 3:
                eps = 1e-9
                beta_g = (
                    1 / eps * (self.beta - 1) / (self.beta**self.iterations_outer - 1)
                )
                tau_tp1 = beta_g * self.beta**t

            theta_t = theta_tp1
            lambda_t = lambda_tp1
            tau_t = tau_tp1

            # log things
            pi_t = jax.nn.softmax(theta_t, axis=1)
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            taus.append(tau_t)
            aug_lag_grad_norms.append(
                jnp.linalg.norm(
                    self.grad_aug_lag(theta_t, lambda_t, tau_t, env, policy)
                ).item()
            )
            lambdas.append(lambda_t.item())
            gradient_evaluations_per_iteration.append(self.iterations_inner)

        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "taus": taus,
            "aug_lag_grad_norms": aug_lag_grad_norms,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = jax.nn.softmax(theta_t)

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
