import tqdm
from algos.base_algo import Algo
import time
from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp

from environments.tabular_env import TabularEnv
from flax import struct


@jax.jit
def projection_simplex_jax(V: jnp.ndarray, z: float = 1.0) -> jnp.ndarray:
    """
    Projects each row of V onto the probability simplex of radius z.
    V: shape (batch_size, n_features)
    z: scalar simplex "sum" (default 1.0)
    """
    batch_size, n_features = V.shape

    # Sort each row in descending order
    U = jnp.sort(V, axis=1)[:, ::-1]

    # Vector of z's for each row
    z_vec = jnp.full((batch_size,), z)

    # Cumulative sum minus z
    cssv = jnp.cumsum(U, axis=1) - z_vec[:, None]

    # 1, 2, ..., n_features
    ind = jnp.arange(1, n_features + 1)

    # Condition for each coordinate
    cond = U - cssv / ind > 0

    # Number of positive entries per row (rho)
    rho = jnp.sum(cond, axis=1)

    # Safety: avoid division by zero in pathological cases
    rho_safe = jnp.maximum(rho, 1)

    # Compute theta for each row
    idx = jnp.arange(batch_size)
    theta = cssv[idx, rho_safe - 1] / rho_safe

    # Final projection
    return jnp.maximum(V - theta[:, None], 0.0)


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


class ALMProjectedPQA(Algo):
    init_lambda: float
    init_tau: float
    eta: float  # policy step-size
    zeta: float  # surrogate step_size
    features: jnp.array
    iterations_alm: int = struct.field(pytree_node=False)  # T
    iterations_projected_pqa: int = struct.field(pytree_node=False)  # K
    iterations_surrogate: int = struct.field(pytree_node=False)  # N

    @classmethod
    def create(
        cls,
        init_lambda,
        init_tau,
        eta,
        zeta,
        features,
        iterations_alm,
        iterations_projected_pqa,
        iterations_surrogate,
    ):
        return cls(
            init_lambda,
            init_tau,
            eta,
            zeta,
            features,
            iterations_alm,
            iterations_projected_pqa,
            iterations_surrogate,
        )

    def alm_pqa(self, env: TabularEnv, policy, theta_t, lambda_t, tau_t):
        # Need to compute Gamma_pi in the functional space
        def q_Gamma_pi(pi, lmbda, tau):
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
            pi_t = policy(theta_inner)
            unprojected_pi_thalf = pi_t + self.eta * q_Gamma_pi(pi_t, lambda_t, tau_t)
            pi_thalf = projection_simplex_jax(unprojected_pi_thalf)

            d_pi_t = env.calc_dpi(pi_t)

            def surrogate_loss(theta):
                log_pi_theta = policy.log(theta)
                return jnp.einsum("s,sa,sa->", d_pi_t, pi_thalf, -log_pi_theta)

            grad_surrogate = jax.grad(surrogate_loss)

            theta_m = theta_inner
            for _ in range(self.iterations_surrogate):
                # zeta = armijo_line_search(surrogate_loss, grad_surrogate, theta_m, 10_000, 0.5, 0.9)
                theta_m = theta_m - self.zeta * grad_surrogate(theta_m)
            return theta_m

        theta_k = theta_t
        for _ in range(self.iterations_projected_pqa):
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

        policy = Policy(self.features)
        theta_t = jnp.zeros(self.features.shape[-1])
        lambda_t = self.init_lambda
        tau_t = self.init_tau

        pi_t = policy(theta_t)
        v_r = env.objective(pi_t)
        v_c = env.constraint(pi_t)

        v_r_pis.append(v_r.item())
        v_c_pis.append(v_c.item())
        aug_lag_grad_norms.append(
            # jnp.linalg.norm(
            #     self.grad_aug_lag(theta_t, lambda_t, tau_t, env, policy)
            # ).item()
            -1.0,
        )
        taus.append(tau_t)
        lambdas.append(lambda_t)

        start_time = time.time()

        for t in tqdm.tqdm(range(self.iterations_alm), desc="T", leave=False):
            theta_tp1 = self.alm_pqa(env, policy, theta_t, lambda_t, tau_t)

            pi_tp1 = policy(theta_tp1)
            vc_tp1 = env.rho @ env.calc_vpi(pi_tp1, env.c)
            z_tp1 = jnp.maximum(vc_tp1 - env.b - lambda_t / tau_t, 0)
            lambda_tp1 = lambda_t - tau_t / 2 * (vc_tp1 - env.b - z_tp1)

            theta_t = theta_tp1
            lambda_t = lambda_tp1

            # log things
            pi_t = policy(theta_t)
            v_r = env.objective(pi_t)
            v_c = env.constraint(pi_t)

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            taus.append(tau_t)
            aug_lag_grad_norms.append(
                # jnp.linalg.norm(
                #     self.grad_aug_lag(theta_t, lambda_t, tau_t, env, policy)
                # ).item()
                -1.0
            )
            lambdas.append(lambda_t.item())
            gradient_evaluations_per_iteration.append(self.iterations_projected_pqa)

            # if verbose:
            #     sub_opt_gap = env.objective(pi_star) - v_r
            #     constraint_violations = env.b - v_c
            #     print(f'T:{t} sub-opt: {sub_opt_gap:05f} cv: {constraint_violations:05f} lambda_t:{lambda_t:0.4f} norm: {aug_lag_grad_norms[-1]:0.4f}')

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
