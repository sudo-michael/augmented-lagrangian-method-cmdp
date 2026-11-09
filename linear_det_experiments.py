import os
import time
from typing import Any
import jax
import jax.numpy as jnp
from contextlib import redirect_stdout
import numpy as np
from tile3 import IHT, tiles


from algos.projected_pqa import ALMProjectedPQA
from algos.linear_npg_pd import LinearNPGPD
from algos.reload import LinearReLoad
from algos.ppo_lag import LinearPPOLag
from environments.constrained_cliffworld import ConstrainedCliffWorld
from environments.constrained_deep_sea_treasure import ConstrainedDeepSeaTreasure
from environments.tabular_env import TabularEnv
from experiment_grid import ExperimentGrid

# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platforms", "cpu")

# params_100 = {
#     "features_one_hot": [False],
#     "features_iht_size": [100],
#     "features_num_tilings": [6],
#     "features_tiling_size": [3],
# }

# params_60_4_2 = {
#     "features_one_hot": [False],
#     "features_iht_size": [60],
#     "features_num_tilings": [4],
#     "features_tiling_size": [2],
# }

# params_60_5_2 = {
#     "features_one_hot": [False],
#     "features_iht_size": [60],
#     "features_num_tilings": [5],
#     "features_tiling_size": [2],
# }

# params_60_3_3 = {
#     "features_one_hot": [False],
#     "features_iht_size": [60],
#     "features_num_tilings": [3],
#     "features_tiling_size": [3],
# }

# params_60_6_2 = {
#     "features_one_hot": [False],
#     "features_iht_size": [60],
#     "features_num_tilings": [6],
#     "features_tiling_size": [2],
# }


class TabularTileCoding:
    def __init__(self, S: int, iht_size: int, num_tilings: int, tiling_size: int = 3):
        """
        Tile-code a discrete tabular state by treating it as a 1D continuous coordinate.
        Returns a 'tiles' for a `(s, a)` pair whose intergers are indices in [0, int_size).
        The 'tiles' can be treated as a sparse binary feature vector.

        Arguments:
            S:
                Number of states in environment. Used only to define bounds [0, S−1] for scaling.

            int_size (feature dimension d):
                Total number of features available.

            num_tilings:
                Number of overlapping grids (tilings). This is also the number of active features
                per (s,a). Larger num_tilings -> smoother / more redundant representation, but more
                compute per step.

            tiling_size:
                Scale factor between each tile. It controls how “spread out” the S discrete states are in the tile
                coder’s 1D continuous space.

                - small tiling_size -> states squished together -> more generalization
                - large tiling_size -> states spread apart -> less generalization
        """
        self.state_bounds_min = 0
        self.state_bounds_max = S - 1

        self.d = iht_size
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.tiling_size = tiling_size

    def tile(self, state: int, action: int) -> list[int]:
        """
        Return sparse feature indices for φ(state, action).

        Output:
          A Python list of length num_tilings. Example: [12, 7, 55, 2, 31]
          Interpret as: φ_i(s,a)=1 for i in that list, else 0.

        Important intuition:
          - We put action in `ints=[action]` so different actions do NOT share features.
          - We treat the tabular state index as a float x, so nearby state indices can share tiles.
        """

        normalized_state = (
            state * self.tiling_size / (self.state_bounds_max - self.state_bounds_min)
        )

        return tiles(self.iht, self.num_tilings, [normalized_state], [action])


def create_features(S: int, A: int, tile_coder: TabularTileCoding) -> np.ndarray:
    d = tile_coder.d
    features = np.zeros((S, A, d))
    for s in range(S):
        for a in range(A):
            tiles = tile_coder.tile(s, a)
            sparse_features = np.zeros(d)
            sparse_features[tiles] = 1
            features[s, a, :] = sparse_features
    return features


def create_one_hot_features(S: int, A: int) -> np.ndarray:
    features = np.zeros((S, A, S * A))
    for s in range(S):
        for a in range(A):
            idx = s * A + a
            features[s, a, idx] = 1.0
    return features


def make_env(env_name):
    if env_name == "ConstrainedCliffWorld":
        np_env = ConstrainedCliffWorld()
        pi_star = np_env.solve_dual()
    elif env_name == "ConstrainedDeepSeaTreasure":
        np_env = ConstrainedDeepSeaTreasure()
        pi_star = np_env.solve_dual()
    else:
        raise AssertionError(f"Unknown environment_name: {env_name}")

    env = TabularEnv.create(
        np_env.P,
        np_env.r,
        np_env.c,
        np_env.b,
        np_env.rho,
        np_env.gamma,
    )

    return env, jnp.array(pi_star)


def train(
    experiment_id: str,
    log_dir: str,
    algo: str,
    env_id: str,
    algo_params: dict[str, Any],
    seed: int,
    verbose: bool,
):
    print(f"{experiment_id} is training...")
    terminal_log_name = f"seed{seed}_terminal.log"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if f"{seed}_output.pickle" in os.listdir(log_dir):
        print(
            f"Experiment found in {log_dir} with {seed=} has already been ran",
        )
        return None

    env, pi_star = make_env(env_id)

    features_one_hot = algo_params.pop("features_one_hot")
    iht_size = algo_params.pop("features_iht_size")
    num_tilings = algo_params.pop("features_num_tilings")
    tiling_size = algo_params.pop("features_tiling_size")
    if features_one_hot:
        features = create_one_hot_features(env.S, env.A)
    else:
        tile_coder = TabularTileCoding(
            S=env.S, iht_size=iht_size, num_tilings=num_tilings, tiling_size=tiling_size
        )
        features = create_features(env.S, env.A, tile_coder)

    if algo == "ALM-PPQA":
        algo = ALMProjectedPQA.create(**algo_params, features=features)
    elif algo == "NPG-RPD":
        algo = LinearNPGPD.create(**algo_params, features=features)
    elif algo == "ReLOAD":
        algo = LinearReLoad.create(**algo_params, features=features)
    elif algo == "PPOLag":
        algo = LinearPPOLag.create(**algo_params, features=features)
    else:
        raise AssertionError("Unknown Algorithm")

    with open(
        os.path.join(
            log_dir,
            terminal_log_name,
        ),
        "w",
    ) as f_out:
        if verbose:
            return algo.train(
                env=env,
                env_id=env_id,
                seed=seed,
                pi_star=pi_star,
                log_dir=log_dir,
                verbose=True,
            )
        with redirect_stdout(f_out):
            return algo.train(
                env=env,
                env_id=env_id,
                seed=seed,
                pi_star=pi_star,
                log_dir=log_dir,
                verbose=True,
            )


def alm_d_80(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ALM-PPQA": {
                "iterations_alm": [10],
                "iterations_projected_pqa": [100],
                "iterations_surrogate": [250],
                "init_lambda": [0.0],
                "init_tau": [10],
                "eta": [0.0001, 0.001, 0.1, 1.0, 10],
                # "eta": [0.1],
                "zeta": [0.0001, 0.001, 0.1, 1.0, 10],
                # "zeta": [0.1],
                "features_one_hot": [False],
                "features_iht_size": [60],
                "features_num_tilings": [5],
                "features_tiling_size": [3],
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def alm(
    experiment_name,
    d=40,
    iterations_alm=20,
    iterations_ppq=250,
    iterations_surrogate=100,
):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ALM-PPQA": {
                "iterations_alm": [iterations_alm],
                "iterations_projected_pqa": [iterations_ppq],
                "iterations_surrogate": [iterations_surrogate],
                "init_lambda": [0.0],
                "init_tau": [10],
                "eta": [0.0001, 0.001, 0.1, 1.0, 10],
                "zeta": [0.0001, 0.001, 0.1, 1.0, 10],
                "features_one_hot": [False],
                "features_iht_size": [d],
                "features_num_tilings": [5],
                "features_tiling_size": [3],
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def alm_d(experiment_name, features):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ALM-PPQA": {
                "iterations_alm": [10],
                "iterations_projected_pqa": [100],
                "iterations_surrogate": [100],
                "init_lambda": [0.0],
                "init_tau": [10],
                "eta": [0.0001, 0.001, 0.1, 1.0, 10],
                "zeta": [0.0001, 0.001, 0.1, 1.0, 10],
                **features,
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def npg_pd_d(experiment_name, features):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "NPG-RPD": {
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "tau": [0.001, 0.01, 0.1],
                "eta_primal": [0.0001, 0.001, 0.1],
                "eta_dual": [0.0001, 0.001, 0.1],
                **features,
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def ppo_lag_d(experiment_name, features):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "PPOLag": {
                "iterations_outer": [1000],
                "iterations_inner": [250],
                "init_lambda": [0.0],
                "eta_primal": [0.0001, 0.001, 0.1, 0.5, 1.0],
                "eta_dual": [0.0001, 0.001, 0.1],
                "clip": [0.1, 0.2, 0.3],
                **features,
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=24, verbose=True)
    print(f"{experiment_name} is finished")


def reload_d(experiment_name, features):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ReLOAD": {
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "eta_primal": [0.0001, 0.001, 0.1, 1.0],
                "eta_dual": [0.0001, 0.001, 0.1, 1.0],
                **features,
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def debug(
    experiment_id: str,
    log_dir: str,
    algo: str,
    env_id: str,
    algo_params: dict[str, Any],
    seed: int,
    verbose: bool,
):
    env, pi_star = make_env(env_id)

    features_one_hot = algo_params.pop("features_one_hot")
    iht_size = algo_params.pop("features_iht_size")
    num_tilings = algo_params.pop("features_num_tilings")
    tiling_size = algo_params.pop("features_tiling_size")
    if features_one_hot:
        features = create_one_hot_features(env.S, env.A)
    else:
        tile_coder = TabularTileCoding(
            S=env.S, iht_size=iht_size, num_tilings=num_tilings, tiling_size=tiling_size
        )
        features = create_features(env.S, env.A, tile_coder)

    # algo = LinearNPGPD.create(**algo_params, features=features)
    algo = LinearPPOLag.create(**algo_params, features=features)
    algo.train(
        env=env,
        env_id=env_id,
        seed=seed,
        pi_star=pi_star,
        log_dir=log_dir,
        verbose=True,
    )


if __name__ in "__main__":
    tik = time.time()
    params_one_hot = {
        "features_one_hot": [True],
        "features_iht_size": [0],
        "features_num_tilings": [0],
        "features_tiling_size": [0],
    }

    params_40 = {
        "features_one_hot": [False],
        "features_iht_size": [40],
        "features_num_tilings": [4],
        "features_tiling_size": [1],
    }

    params_60 = {
        "features_one_hot": [False],
        "features_iht_size": [60],
        "features_num_tilings": [4],
        "features_tiling_size": [3],
    }

    params_80 = {
        "features_one_hot": [False],
        "features_iht_size": [80],
        "features_num_tilings": [6],
        "features_tiling_size": [5],
    }

    ppo_lag_d("ppo_lag_one_hot_1000", params_one_hot)
    ppo_lag_d("ppo_lag_d_40_1000", params_40)
    ppo_lag_d("ppo_lag_d_60_1000", params_60)
    ppo_lag_d("ppo_lag_d_80_1000", params_80)

    npg_pd_d("npg_pd_linear_d_40_1000", params_40)
    npg_pd_d("npg_pd_linear_d_60_1000", params_60)
    npg_pd_d("npg_pd_linear_d_80_1000", params_80)

    reload_d("reload_linear_d_40_1000", params_40)
    reload_d("reload_linear_d_60_1000", params_60)
    reload_d("reload_linear_d_80_1000", params_80)

    alm_d("alm_linear_d_40_1000", params_40)
    alm_d("alm_linear_d_60_1000", params_60)
    alm_d("alm_linear_d_80_1000", params_80)

    tok = time.time()
    print(f"Total benchmark time: {tok - tik}")
