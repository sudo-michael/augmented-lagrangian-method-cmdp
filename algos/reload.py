import tqdm
from algos.base_algo import Algo
import time
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv


# class ReLoad(Algo):
#     iterations_outer: int
#     init_lambda: float
#     eta_primal: float
#     eta_dual: float

#     @classmethod
#     def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual):
#         return cls(iterations_outer, init_lambda, eta_primal, eta_dual)

#     def train(
#         self,
#         log_dir: str,
#         env: TabularEnv,
#         env_id: str,
#         pi_star: Optional[jnp.ndarray] = None,
#         seed: Optional[int] = 0,
#         verbose: Optional[bool] = False,
#     ):
#         key = jax.random.PRNGKey(seed)
#         v_r_pis = []
#         v_c_pis = []
#         lambdas = []
#         gradient_evaluations_per_iteration = []

#         lambda_t = self.init_lambda
#         lambda_t_m_half = self.init_lambda

#         pi_t = jax.nn.softmax(jnp.zeros_like(env.r), axis=1)
#         pi_t_m_half = jax.nn.softmax(jnp.zeros_like(env.r), axis=1)

#         v_r = env.objective(pi_t)
#         v_c = env.constraint(pi_t)

#         v_r_pis.append(v_r.item())
#         v_c_pis.append(v_c.item())
#         lambdas.append(lambda_t)

#         start_time = time.time()

#         for _ in range(self.iterations_outer):
#             # half step
#             q_r_t_m_half = env.calc_qpi(pi_t_m_half, env.r)
#             q_c_t_m_half = env.calc_qpi(pi_t_m_half, env.c)

#             # In ReLOAD, their convention is for minimization, so some signs are flipped here
#             q_t_m_half = +q_r_t_m_half + q_c_t_m_half * lambda_t_m_half

#             pi_t_half = jax.nn.softmax(
#                 jnp.log(pi_t) + q_t_m_half / self.eta_primal, axis=1
#             )
#             lambda_t_half = max(
#                 lambda_t - self.eta_dual * (env.constraint(pi_t_m_half) - env.b), 0
#             )

#             q_r_t_half = env.calc_qpi(pi_t_half, env.r)
#             q_c_t_half = env.calc_qpi(pi_t_half, env.c)
#             q_t_half = +q_r_t_half + q_c_t_half * lambda_t_half

#             # full step
#             pi_tp1 = jax.nn.softmax(jnp.log(pi_t) + q_t_half / self.eta_primal, axis=1)
#             lambda_tp1 = max(
#                 lambda_t - self.eta_dual * (env.constraint(pi_t_half) - env.b), 0
#             )

#             # update variables
#             pi_t_m_half = pi_t_half
#             lambda_t_m_half = lambda_t_half

#             pi_t = pi_tp1
#             lambda_t = lambda_tp1

#             # log
#             v_r = env.objective(pi_t)
#             v_c = env.constraint(pi_t)

#             v_r_pis.append(v_r.item())
#             v_c_pis.append(v_c.item())
#             lambdas.append(lambda_t)
#             gradient_evaluations_per_iteration.append(2)
#         end_time = time.time()

#         log_info = {
#             "v_r_pis": v_r_pis,
#             "v_c_pis": v_c_pis,
#             "lambas": lambdas,
#             "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
#         }
#         pi_T = pi_t

#         self.save_log(
#             log_info=log_info,
#             log_dir=log_dir,
#             env=env,
#             env_id=env_id,
#             pi_T=pi_T,
#             pi_star=pi_star,
#             seed=seed,
#             elapsed_time=end_time - start_time,
#         )

#         return True


class ReLoad(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual)

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

        lambda_t = self.init_lambda
        lambda_tm1 = self.init_lambda

        pi_t = jax.nn.softmax(jnp.zeros_like(env.r), axis=1)
        pi_tm1 = jax.nn.softmax(jnp.zeros_like(env.r), axis=1)

        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lambda_t)

        start_time = time.time()

        for _ in range(self.iterations_outer):
            q_r_t = env.calc_qpi(pi_t, env.r)
            q_c_t = env.calc_qpi(pi_t, env.c)
            q_t = +q_r_t + q_c_t * lambda_t

            q_r_tm1 = env.calc_qpi(pi_tm1, env.r)
            q_c_tm1 = env.calc_qpi(pi_t, env.c)
            q_tm1 = +q_r_tm1 + q_c_tm1 * lambda_tm1

            # full step
            pi_tp1 = jax.nn.softmax(
                jnp.log(pi_t) + (2 * q_t - q_tm1) / self.eta_primal, axis=1
            )
            lambda_tp1 = max(
                lambda_t
                - self.eta_dual
                * (2 * env.constraint(pi_t) - env.constraint(pi_tm1) - env.b),
                0,
            )

            pi_tm1 = pi_t
            lambda_tm1 = lambda_t

            pi_t = pi_tp1
            lambda_t = lambda_tp1

            # log
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lambda_t)
            gradient_evaluations_per_iteration.append(2)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = pi_t

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


# class LinearReLoad(Algo):
#     iterations_outer: int
#     init_lambda: float
#     eta_primal: float
#     eta_dual: float
#     features: jnp.array

#     @classmethod
#     def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual, features):
#         return cls(iterations_outer, init_lambda, eta_primal, eta_dual, features)

#     def solve_compatible_regression(self, env: TabularEnv, theta, q, ridge=0.001):
#         pi = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta), axis=1)
#         d_pi = env.calc_dpi(pi)
#         mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

#         def log_pi(theta):
#             return jax.nn.log_softmax(
#                 jnp.einsum("sad,d->sa", self.features, theta), axis=1
#             )

#         psi = jax.jacrev(log_pi)(theta)

#         y = jnp.einsum("sa,sa,sad->d", mu_pi, q, psi)
#         K = jnp.einsum("sa,sad,sam->dm", mu_pi, psi, psi)

#         # y = jnp.einsum("sa,sa,sad->d", mu_pi, q, self.features)
#         # K = jnp.einsum("sa,sad,sam->dm", mu_pi, self.features, self.features)
#         # w = np.linalg.lstsq(a=K, b=y)[0]
#         try:
#             w = np.linalg.solve(K, y)
#         except np.linalg.LinAlgError:
#             w = np.linalg.lstsq(a=K + ridge * np.eye(K.shape[0]), b=y)[0]

#         # qr = jnp.einsum("sad,d->sa", self.features, w)
#         # breakpoint()

#         return w

#     def train(
#         self,
#         log_dir: str,
#         env: TabularEnv,
#         env_id: str,
#         pi_star: Optional[jnp.ndarray] = None,
#         seed: Optional[int] = 0,
#         verbose: Optional[bool] = False,
#     ):
#         key = jax.random.PRNGKey(seed)
#         v_r_pis = []
#         v_c_pis = []
#         lambdas = []
#         gradient_evaluations_per_iteration = []

#         lambda_t = self.init_lambda
#         lambda_t_m_half = self.init_lambda

#         d = self.features.shape[-1]
#         theta_t = jnp.zeros(d)
#         pi_t = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta_t), axis=1)
#         pi_t_m_half = jax.nn.softmax(
#             jnp.einsum("sad,d->sa", self.features, theta_t), axis=1
#         )

#         v_r = env.objective(pi_t)
#         v_c = env.constraint(pi_t)

#         v_r_pis.append(v_r.item())
#         v_c_pis.append(v_c.item())
#         lambdas.append(lambda_t)

#         start_time = time.time()

#         for t in range(self.iterations_outer):
#             # half step
#             q_r_t_m_half = env.calc_qpi(pi_t_m_half, env.r)
#             q_c_t_m_half = env.calc_qpi(pi_t_m_half, env.c)

#             # In ReLOAD, their convention is for minimization, so some signs are flipped here
#             q_t_m_half = +q_r_t_m_half + q_c_t_m_half * lambda_t_m_half

#             w = self.solve_compatible_regression(env, theta_t, q_t_m_half)
#             theta_t_half = theta_t + 1/self.eta_primal * w
#             pi_t_half = jax.nn.softmax(
#                 jnp.einsum("sad,d->sa", self.features, theta_t_half), axis=1
#             )
#             lambda_t_half = max(
#                 lambda_t - self.eta_dual * (env.constraint(pi_t_m_half) - env.b), 0
#             )

#             q_r_t_half = env.calc_qpi(pi_t_half, env.r)
#             q_c_t_half = env.calc_qpi(pi_t_half, env.c)
#             q_t_half = +q_r_t_half + q_c_t_half * lambda_t_half

#             # full step
#             w_t_half = self.solve_compatible_regression(env, theta_t, q_t_half)
#             theta_tp1 = theta_t + 1/self.eta_primal * w_t_half
#             pi_tp1 = jax.nn.softmax(
#                 jnp.einsum("sad,d->sa", self.features, theta_tp1), axis=1
#             )

#             lambda_tp1 = max(
#                 lambda_t - self.eta_dual * (env.constraint(pi_t_half) - env.b), 0
#             )

#             # update variables
#             pi_t_m_half = pi_t_half
#             lambda_t_m_half = lambda_t_half

#             pi_t = pi_tp1
#             theta_t = theta_tp1
#             lambda_t = lambda_tp1

#             # log
#             v_r = env.objective(pi_t)
#             v_c = env.constraint(pi_t)

#             v_r_pis.append(v_r.item())
#             v_c_pis.append(v_c.item())
#             lambdas.append(lambda_t)
#             gradient_evaluations_per_iteration.append(2)

#             if verbose:
#                 sub_opt_gap = env.objective(pi_star) - v_r
#                 constraint_violations = env.b - v_c
#                 print(f'T:{t} sub-opt: {sub_opt_gap:05f} cv: {constraint_violations:05f} lambda_t:{lambda_t:0.4f}')
#         end_time = time.time()

#         log_info = {
#             "v_r_pis": v_r_pis,
#             "v_c_pis": v_c_pis,
#             "lambas": lambdas,
#             "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
#         }
#         pi_T = pi_t

#         self.save_log(
#             log_info=log_info,
#             log_dir=log_dir,
#             env=env,
#             env_id=env_id,
#             pi_T=pi_T,
#             pi_star=pi_star,
#             seed=seed,
#             elapsed_time=end_time - start_time,
#         )

#         return True


class LinearReLoad(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float
    features: jnp.array

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual, features):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual, features)

    def solve_compatible_regression(self, env: TabularEnv, theta, q, ridge=0.001):
        pi = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta), axis=1)
        d_pi = env.calc_dpi(pi)
        mu_pi = jnp.einsum("sa, s->sa", pi, d_pi)

        def log_pi(theta):
            return jax.nn.log_softmax(
                jnp.einsum("sad,d->sa", self.features, theta), axis=1
            )

        psi = jax.jacrev(log_pi)(theta)

        y = jnp.einsum("sa,sa,sad->d", mu_pi, q, psi)
        K = jnp.einsum("sa,sad,sam->dm", mu_pi, psi, psi)
        try:
            w = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a=K + ridge * np.eye(K.shape[0]), b=y)[0]

        # qr = jnp.einsum("sad,d->sa", self.features, w)
        # breakpoint()

        return w

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

        lambda_t = self.init_lambda
        lambda_tm1 = self.init_lambda

        d = self.features.shape[-1]
        theta_t = jnp.zeros(d)
        pi_t = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta_t), axis=1)
        pi_tm1 = jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta_t), axis=1)

        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lambda_t)

        start_time = time.time()

        for t in range(self.iterations_outer):
            q_r_tm1 = env.calc_qpi(pi_tm1, env.r)
            q_c_tm1 = env.calc_qpi(pi_tm1, env.c)

            # In ReLOAD, their convention is for minimization, so some signs are flipped here
            q_tm1 = +q_r_tm1 + q_c_tm1 * lambda_tm1

            q_r_t = env.calc_qpi(pi_t, env.r)
            q_c_t = env.calc_qpi(pi_t, env.c)

            # In ReLOAD, their convention is for minimization, so some signs are flipped here
            q_t = +q_r_t + q_c_t * lambda_t

            w = self.solve_compatible_regression(env, theta_t, 2 * q_t - q_tm1)
            theta_tp1 = theta_t + 1 / self.eta_primal * w
            pi_tp1 = jax.nn.softmax(
                jnp.log(pi_t) + jnp.einsum("sad,d->sa", self.features, theta_tp1),
                axis=1,
            )
            lambda_tp1 = max(
                lambda_t
                - self.eta_dual
                * (2 * env.constraint(pi_tp1) - env.constraint(pi_t) - env.b),
                0,
            )

            # update variables
            pi_tm1 = pi_t
            lambda_tm1 = lambda_t

            pi_t = pi_tp1
            theta_t = theta_tp1
            lambda_t = lambda_tp1

            # log
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lambda_t)
            gradient_evaluations_per_iteration.append(2)

            if verbose:
                sub_opt_gap = env.objective(pi_star) - v_r
                constraint_violations = env.b - v_c
                print(
                    f"T:{t} sub-opt: {sub_opt_gap:05f} cv: {constraint_violations:05f} lambda_t:{lambda_t:0.4f}"
                )
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = pi_t

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
