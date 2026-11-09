from functools import partial

import tqdm
from algos.base_algo import Algo
import time
from typing import Optional
import numpy as np

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


class ALMPG(Algo):
    iterations_outer: int
    iterations_inner: int
    init_lambda: float
    init_tau: float
    beta: float
    max_tau: float
    eta: float
    tau_stratergy: int
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

        # original augmented lagrangian (general utility) reward
        # mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)
        # def F(mu_pi):
        #     vr_pi = jnp.dot(mu_pi.flatten(), env.r.flatten()) / (1 - env.gamma)
        #     vc_pi = jnp.dot(mu_pi.flatten(), env.c.flatten()) / (1 - env.gamma)

        #     return vr_pi + (tau / 2) * (
        #         -(jnp.minimum(0, -lmbda / tau + (vc_pi - env.b)) ** 2)
        #         + lmbda**2 / tau**2 #     )
        # Gamma_pi = jax.grad(F)(mu_pi)

        # translated above function to using if-statement
        # if cons >= env.b + lmbda / tau:
        #     Gamma_pi = env.r
        # else:
        #     Gamma_pi = env.r + lmbda * env.c - tau * (cons - env.b) * env.c

        # JIT-compatible version
        cons = env.constraint(pi)
        condition = cons >= env.b + lmbda / tau
        Gamma_pi = jnp.where(
            condition, env.r, env.r + lmbda * env.c - tau * (cons - env.b) * env.c
        )

        q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)

        d_pi = env.calc_dpi(pi)

        @jax.grad
        def pg(theta):
            return jnp.dot(
                d_pi,
                jnp.einsum("sa, sa->s", q_Gamma_pi, policy(theta)),
            )

        return pg(theta)

    @jax.jit
    def calc_q_Gamma_pi(self, theta, lmbda, tau, env, policy):
        pi = policy(theta)

        cons = env.constraint(pi)
        condition = cons >= env.b + lmbda / tau
        Gamma_pi = jnp.where(
            condition, env.r, env.r + lmbda * env.c - tau * (cons - env.b) * env.c
        )

        q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)
        return q_Gamma_pi

    def alm_pqa(self, env: TabularEnv, policy, theta_t, lambda_t, tau_t, t, pi_star):
        def projection_simplex(V, z=1):
            # https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
            n_features = V.shape[1]
            U = np.sort(V, axis=1)[:, ::-1]
            z = np.ones(len(V)) * z
            cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
            ind = np.arange(n_features) + 1
            cond = U - cssv / ind > 0
            rho = np.count_nonzero(cond, axis=1)
            theta = cssv[np.arange(len(V)), rho - 1] / rho
            return np.maximum(V - theta[:, np.newaxis], 0)

        @jax.jit
        def q_Gamma_pi(theta, lmbda, tau):
            pi = policy(theta)
            d_pi = env.calc_dpi(pi)
            mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

            def F(mu_pi):
                vr_pi = jnp.dot(mu_pi.flatten(), env.r.flatten()) / (1 - env.gamma)
                vc_pi = jnp.dot(mu_pi.flatten(), env.c.flatten()) / (1 - env.gamma)

                return vr_pi + (tau / 2) * (
                    -(jnp.minimum(0, vc_pi - env.b - lmbda / tau) ** 2)
                    + lmbda**2 / tau**2
                )

            Gamma_pi = jax.grad(F)(mu_pi)

            q_Gamma_pi = env.calc_qpi(pi, Gamma_pi)

            return q_Gamma_pi

        def update(theta_inner, lambda_t, tau_t):
            df = partial(q_Gamma_pi, lmbda=lambda_t, tau=tau_t)
            theta_inner += self.eta * df(theta_inner)
            return projection_simplex(theta_inner)

        theta_k = theta_t
        for _ in range(self.iterations_inner):
            theta_k = update(theta_k, lambda_t, tau_t)

        return jnp.array(theta_k)

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

        policy = Policy(
            num_actions=env.A, functional_representation=self.functional_representation
        )
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
        def constraint_minus_b(theta):
            pi = policy(theta)
            vc = env.rho @ env.calc_vpi(pi, env.c)
            return vc - env.b

        start_time = time.time()

        for t in tqdm.tqdm(range(self.iterations_outer), desc="T", leave=False):
            theta_tp1 = self.alm_pqa(env, policy, theta_t, lambda_t, tau_t, t, pi_star)

            pi = policy(theta_tp1)

            vc_tp1 = env.rho @ env.calc_vpi(pi, env.c)

            z_tp1 = jnp.maximum(vc_tp1 - env.b - lambda_t / tau_t, 0)
            lambda_tp1 = lambda_t - tau_t / 2 * (vc_tp1 - env.b - z_tp1)

            # 3. increase tau
            tau_tp1 = tau_t

            theta_t = theta_tp1
            lambda_t = lambda_tp1
            tau_t = tau_tp1

            # log things
            pi_t = policy(theta_t)
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

            if verbose:
                sub_opt_gap = env.objective(pi_star) - v_r
                constraint_violations = env.b - v_c
                print(
                    f"T:{t} sub-opt: {sub_opt_gap:05f} cv: {constraint_violations:05f} lambda_t:{lambda_t:0.4f} norm: {aug_lag_grad_norms[-1]:0.4f}"
                )

        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "taus": taus,
            "aug_lag_grad_norms": aug_lag_grad_norms,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = policy(theta_t)

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
