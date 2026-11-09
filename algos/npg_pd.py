import numpy as np
import tqdm
from algos.base_algo import Algo
import time
from typing import Optional

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv


class NPGPD(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float
    tau: float

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual, tau):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual, tau)

    @jax.jit
    def update(self, env, theta_t, lambda_t):
        pi_t = jax.nn.softmax(theta_t, axis=1)
        q_r = env.calc_qpi(pi_t, env.r)
        q_c = env.calc_qpi(pi_t, env.c)

        psi = -jax.nn.log_softmax(theta_t, axis=1)
        q_psi = env.calc_qpi(pi_t, psi)
        lag = q_r + lambda_t * q_c + self.tau * q_psi

        theta_tp1 = theta_t + self.eta_primal * lag

        vc = env.rho @ env.calc_vpi(pi_t, env.c)
        lambda_tp1 = jnp.clip(
            lambda_t - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
        )
        return theta_tp1, lambda_tp1

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
        lambda_t = self.init_lambda

        pi_t = jax.nn.softmax(theta_t, axis=1)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lambda_t)

        start_time = time.time()

        for _ in range(self.iterations_outer):
            theta_t, lambda_t = self.update(env, theta_t, lambda_t)

            # log
            pi_t = jax.nn.softmax(theta_t, axis=1)
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lambda_t.item())
            gradient_evaluations_per_iteration.append(2)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = jax.nn.softmax(theta_t, axis=1)

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
