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
    features: jnp.array

    @classmethod
    def create(cls, features):
        return cls(features=features)

    @jax.jit
    def __call__(self, theta):
        return jax.nn.softmax(jnp.einsum("sad,d->sa", self.features, theta), axis=1)

    @jax.jit
    def log(self, theta):
        return jax.nn.log_softmax(jnp.einsum("sad,d->sa", self.features, theta), axis=1)


class LinearPPOLag(Algo):
    iterations_outer: int
    iterations_inner: int
    init_lambda: float
    eta_primal: float
    eta_dual: float
    clip: float
    features: jnp.array

    @classmethod
    def create(
        cls,
        iterations_outer,
        iterations_inner,
        init_lambda,
        eta_primal,
        eta_dual,
        clip,
        features,
    ):
        return cls(
            iterations_outer,
            iterations_inner,
            init_lambda,
            eta_primal,
            eta_dual,
            clip,
            features,
        )

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
        return w

    def update(self, env, policy, theta_t, lambda_t):
        pi_t = policy(theta_t)
        a_r = env.calc_api(pi_t, env.r)
        a_c = env.calc_api(pi_t, env.c)

        adv_lag = a_r + lambda_t * a_c

        d_pi = env.calc_dpi(pi_t)

        mu_pi_old = jnp.einsum("sa, s->sa", pi_t, d_pi)
        logp_old = policy(theta_t)

        def ppo_objective(theta):
            logp = policy.log(theta)
            ratio = jnp.exp(logp - logp_old)
            clipped = jnp.clip(ratio, 1.0 - self.clip, 1.0 + self.clip)

            surr1 = ratio * adv_lag
            surr2 = clipped * adv_lag
            surr = jnp.minimum(surr1, surr2)

            return (mu_pi_old * surr).sum()

        grad_fn = jax.jit(jax.grad(ppo_objective))

        # w = self.solve_compatible_regression(env, theta_t, a_lag)
        theta = theta_t
        for _ in range(self.iterations_inner):
            g = grad_fn(theta)
            theta = theta + self.eta_primal * g

        pi_tp1 = policy(theta)

        # dual update
        vc = env.rho @ env.calc_vpi(pi_tp1, env.c)
        lambda_tp1 = jnp.clip(
            lambda_t - self.eta_dual * (vc - env.b), 0, 1000 / (1 - env.gamma)
        )
        return theta, lambda_tp1

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

        policy = Policy(self.features)
        theta_t = jnp.zeros(self.features.shape[-1])
        lambda_t = self.init_lambda

        pi_t = policy(theta_t)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        lambdas.append(lambda_t)

        start_time = time.time()

        for t in tqdm.tqdm(range(self.iterations_outer), desc="T", leave=False):
            theta_t, lambda_t = self.update(env, policy, theta_t, lambda_t)

            # log things
            pi_t = policy(theta_t)
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lambda_t.item())

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
