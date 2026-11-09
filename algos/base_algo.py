import os
import pickle
import time
from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from environments.tabular_env import TabularEnv


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


class Algo(struct.PyTreeNode):
    def train(
        self,
        log_dir: str,
        env: TabularEnv,
        env_id: str,
        pi_star: Optional[jnp.ndarray] = None,
        seed: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        pass

    def save_log(
        self,
        log_info: str,
        log_dir: str,
        env: TabularEnv,
        env_id: str,
        pi_T: jnp.ndarray,
        pi_star: jnp.ndarray,
        seed: int,
        elapsed_time: float,
    ):
        log_info["seed"] = seed
        log_info["env_id"] = env_id
        log_info["elapsed_time"] = elapsed_time

        # save performance of last policy
        vr_pi_T = env.objective(pi_T)
        vc_pi_T = env.constraint(pi_T)
        log_info["v_r_pi_T"] = vr_pi_T.item()
        log_info["v_c_pi_T"] = vc_pi_T.item()
        log_info["pi_T"] = pi_T

        if pi_star is not None:
            vr_star = env.objective(pi_star)
            vc_star = env.constraint(pi_star)
            log_info["v_r_pistar"] = vr_star.item()
            log_info["v_c_pistar"] = vc_star.item()
            log_info["b"] = env.b

            print("sub-opt-gap", vr_star - vr_pi_T, "cv", env.b - vc_pi_T)

        with open(os.path.join(log_dir, f"{seed}_output.pickle"), "wb") as handle:
            pickle.dump(log_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TestAlgo(Algo):
    mean: float
    noise: float
    iterations_outer: int

    @classmethod
    def create(cls, mean, noise, iterations_outer):
        return cls(mean=mean, noise=noise, iterations_outer=iterations_outer)

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
        start_time = time.time()

        for _ in range(self.iterations_outer):
            key, value_key, cost_key = jax.random.split(key, 3)
            v_r = jax.random.normal(value_key) * self.noise + self.mean
            v_c = jax.random.normal(cost_key) * self.noise + self.mean
            lmbda = 0

            v_r_pis.append(v_r.item())
            v_c_pis.append(v_c.item())
            lambdas.append(lmbda)
            gradient_evaluations_per_iteration.append(1)
        end_time = time.time()

        log_info = {
            "v_r_pis": v_r_pis,
            "v_c_pis": v_c_pis,
            "lambas": lambdas,
            "gradient_evaluations_per_iteration": gradient_evaluations_per_iteration,
        }
        pi_T = jax.nn.softmax(jnp.zeros_like(env.r))

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
