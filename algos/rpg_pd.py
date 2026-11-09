import numpy as np
import tqdm
from algos.base_algo import Algo
import time
from typing import Optional

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv


class RPGPD(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual)

    def projection_simplex(self, V, z=1):
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

        key, theta_key, theta_h_key = jax.random.split(key, 3)
        theta = jnp.ones_like(env.r) / env.A
        theta_h = jax.random.uniform(theta_h_key, shape=env.r.shape)
        lmda = self.init_lambda
        lmda_h = self.init_lambda

        pi = theta

        v_r = env.objective(pi)
        v_c = env.constraint(pi)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lmda_h)

        start_time = time.time()

        for _ in range(self.iterations_outer):
            # optimistic step for (theta, dual)
            pi = theta
            q_r = env.calc_qpi(pi, env.r)
            q_c = env.calc_qpi(pi, env.c)
            q_lag = q_r + lmda * q_c

            theta = self.projection_simplex(theta_h + self.eta_primal * q_lag)
            vc = env.rho @ env.calc_vpi(pi, env.c)
            lmda = jnp.clip(
                lmda_h - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
            )

            # optimistic step for (theta_h, dual_h)
            pi = theta
            q_r = env.calc_qpi(pi, env.r)
            q_c = env.calc_qpi(pi, env.c)
            q_lag = q_r + lmda * q_c

            theta_h = self.projection_simplex(theta_h + self.eta_primal * q_lag)
            lmda_h = jnp.clip(
                lmda_h - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
            )

            # log
            pi_t = theta_h
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lmda_h.item())
            gradient_evaluations_per_iteration.append(4)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = theta_h

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


class StochasticOPGDAPD(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float
    num_trajs: int

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual, num_trajs):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual, num_trajs)

    def projection_simplex(self, V, z=1):
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

        key, theta_key, theta_h_key = jax.random.split(key, 3)
        theta = jnp.ones_like(env.r) / env.A
        theta_h = jax.random.uniform(theta_h_key, shape=env.r.shape)
        lmda = self.init_lambda
        lmda_h = self.init_lambda

        pi = theta

        v_r = env.objective(pi)
        v_c = env.constraint(pi)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lmda_h)

        start_time = time.time()

        def sample_fn(key, theta, state):
            action = jax.random.choice(key, a=env.A, p=theta[state])
            return action

        for _ in tqdm.tqdm(range(self.iterations_outer)):
            # optimistic step for (theta, dual)
            key, optimistic_key = jax.random.split(key)
            pi = theta
            q_r, q_c = env.estimate_q_pi_mc(
                optimistic_key, self.num_trajs, sample_fn, pi
            )
            q_lag = q_r + lmda * q_c

            theta = self.projection_simplex(theta_h + self.eta_primal * q_lag)

            vc = jnp.einsum("sa, sa->s", q_c, pi).mean()

            lmda = jnp.clip(
                lmda_h - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
            )

            # optimistic step for (theta_h, dual_h)
            key, optimistic_h_key = jax.random.split(key)
            pi = theta
            q_r, q_c = env.estimate_q_pi_mc(
                optimistic_h_key, self.num_trajs, sample_fn, pi
            )
            q_lag = q_r + lmda * q_c

            theta_h = self.projection_simplex(theta_h + self.eta_primal * q_lag)
            lmda_h = jnp.clip(
                lmda_h - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
            )

            # log
            pi_t = theta_h
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lmda_h.item())
            gradient_evaluations_per_iteration.append(4)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = theta_h

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
