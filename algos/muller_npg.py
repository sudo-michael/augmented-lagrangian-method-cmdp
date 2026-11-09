import tqdm
from algos.base_algo import Algo
import time
from typing import Optional

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv


class MullerNPGPD(Algo):
    iterations_outer: int
    init_lambda: float
    eta: float
    tau: float

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta, tau):
        return cls(iterations_outer, init_lambda, eta, tau)

    @jax.jit
    def update(self, env, theta_t, lambda_t):
        pi_t = jax.nn.softmax(theta_t, axis=1)
        pi_safe = jnp.clip(pi_t, 1e-12, 1.0)
        entropy = -self.tau * jnp.sum(pi_safe * jnp.log(pi_safe), axis=1, keepdims=True)

        q_r = env.calc_qpi(pi_t, env.r)
        q_c = env.calc_qpi(pi_t, env.c)
        q_lag = q_r + lambda_t * q_c + entropy

        # primal update
        theta_tp1 = theta_t + self.eta * q_lag

        # dual update
        # TODO figure upper bound of slader's condition
        vc = env.rho @ env.calc_vpi(pi_t, env.c)
        lambda_tp1 = jnp.clip(
            (1 - self.eta * self.tau) * lambda_t - self.eta * (vc - env.b),
            0,
            1000 / (1 - env.gamma),
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


class StochasticNPGPD(Algo):
    iterations_outer: int
    init_lambda: float
    eta_primal: float
    eta_dual: float
    num_trajs: int

    @classmethod
    def create(cls, iterations_outer, init_lambda, eta_primal, eta_dual, num_trajs):
        return cls(iterations_outer, init_lambda, eta_primal, eta_dual, num_trajs)

    def update(self, env, theta_t, lambda_t, key, sample_fn):
        q_r, q_c = env.estimate_q_pi_mc(key, self.num_trajs, sample_fn, theta_t)
        # pi_t = jax.nn.softmax(theta_t, axis=1)
        # advantage_r = env.calc_qpi(pi_t, env.r) - env.calc_vpi(pi_t, env.r).reshape(
        #     -1, 1
        # )
        # advantage_c = env.calc_qpi(pi_t, env.c) - env.calc_vpi(pi_t, env.c).reshape(
        #     -1, 1
        # )
        # advantage_lag = advantage_r + lambda_t * advantage_c

        # # primal update
        # theta_tp1 = theta_t + self.eta_primal * advantage_lag

        # # dual update
        # # TODO figure upper bound of slader's condition
        # vc = env.rho @ env.calc_vpi(pi_t, env.c)
        # lambda_tp1 = jnp.clip(
        #     lambda_t - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
        # )
        return self._update_jit(env, theta_t, lambda_t, q_r, q_c)

    def _update_jit(self, env, theta_t, lambda_t, q_r, q_c):
        pi_t = jax.nn.softmax(theta_t, axis=1)
        advantage_r = q_r - jnp.einsum("sa, sa->s", q_r, pi_t).reshape(-1, 1)
        advantage_c = q_c - jnp.einsum("sa, sa->s", q_c, pi_t).reshape(-1, 1)
        advantage_lag = advantage_r + lambda_t * advantage_c

        # primal update
        theta_tp1 = theta_t + self.eta_primal * advantage_lag

        # dual update
        # TODO figure upper bound of slader's condition
        vc = jnp.einsum("sa, sa->s", q_c, pi_t).mean()
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

        # IMPORTANT: the sampling function should only be constructed once
        # otherwise, when estimating Q^\pi, we will need to recompile very often
        # and will cause OOM errors that
        def sample_fn(key, theta, state):
            pi = jax.nn.softmax(theta, axis=1)
            action = jax.random.choice(key, a=env.A, p=pi[state])
            return action

        for t in tqdm.tqdm(range(self.iterations_outer), desc="T", leave=False):
            key, update_key = jax.random.split(key)
            theta_t, lambda_t = self.update(
                env, theta_t, lambda_t, update_key, sample_fn
            )

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
