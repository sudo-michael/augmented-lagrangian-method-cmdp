import os
import time
from typing import Any
import jax
import jax.numpy as jnp
from contextlib import redirect_stdout

from algos.npg_pd import NPGPD
from algos.muller_npg import MullerNPGPD
from algos.ying_npg import YINGNPGPD
from algos.rpg_pd import RPGPD
from algos.reload import ReLoad
from algos.alm_pg import ALMPG
from environments.constrained_cliffworld import ConstrainedCliffWorld
from environments.constrained_deep_sea_treasure import ConstrainedDeepSeaTreasure
from environments.tabular_env import TabularEnv
from experiment_grid import ExperimentGrid

# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platforms", "cpu")


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

    if algo == "ReLOAD":
        algo = ReLoad.create(**algo_params)
    elif algo == "NPG-PD":
        algo = NPGPD.create(**algo_params)
    elif algo == "RPG-PD":
        algo = RPGPD.create(**algo_params)
    elif algo == "Muller-NPG-PD":
        algo = MullerNPGPD.create(**algo_params)
    elif algo == "Ying-NPG-PD":
        algo = YINGNPGPD.create(**algo_params)
    elif algo == "ALM-PQA":
        algo = ALMPG.create(**algo_params)
    else:
        raise AssertionError("unknown algorithm")

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


def npg_pg(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "NPG-PD": {  # 2 gradient evaluation per iteration
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "tau": [0.0],
                "eta_primal": [0.01, 0.1, 1, 10, 100],
                "eta_dual": [0.01, 0.1, 1, 10, 100],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def rpg_pg(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "RPG-PD": {
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "eta_primal": [0.01, 0.1, 1, 10, 100],
                "eta_dual": [0.01, 0.1, 1, 10, 100],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


# baseline experiment
def ogda_pg(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "OGDA-PD": {  # 4 gradient evaluation per iteration
                "iterations_outer": [1000 // 4],
                "init_lambda": [0.0],
                "eta_primal": [0.01, 0.1, 1, 10, 100],
                "eta_dual": [0.01, 0.1, 1, 10, 100],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def muller_npg_pg(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "Muller-NPG-PD": {
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "eta": [0.01, 0.1, 1, 10, 100],
                "tau": [0.001, 0.01, 0.1, 1],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def ying_npg_pg(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "Ying-NPG-PD": {
                "functional_representation": ["direct"],
                "iterations_outer": [100],
                "iterations_inner": [10],
                "init_lambda": [0.0],
                "tau": [0.001, 0.01, 0.1, 1],
                "alpha": [0.001, 0.01, 0.1, 1],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


def reload(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ReLOAD": {
                "iterations_outer": [1000],
                "init_lambda": [0.0],
                "eta_primal": [0.001, 0.01, 0.1, 1],
                "eta_dual": [0.001, 0.01, 0.1, 1],
            },
        },
        "seed": [0],
    }

    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")

def alm_pqa(experiment_name):
    config = {
        "env_id": ["ConstrainedCliffWorld", "ConstrainedDeepSeaTreasure"],
        "algo": {
            "ALM-PQA": {
                "iterations_outer": [10],
                "iterations_inner": [100],
                "functional_representation": ["direct"],
                "algo_inner": ["pqa"],
                "init_tau": [10.0],
                "init_lambda": [0.0],
                "max_tau": [100],
                "beta": [1.0],
                "eta": [0.1, 1.0, 10],
                "tau_stratergy": [None],
            },
        },
        "seed": [0],
    }
    eg = ExperimentGrid(experiment_name, config)
    eg.run(train, num_pool=12, verbose=True)
    print(f"{experiment_name} is finished")


if __name__ in "__main__":
    tik = time.time()
    npg_pg("npg_pd_1000")
    rpg_pg("ding_rpg_pd_1000")
    muller_npg_pg("muller_npg_pd_1000")
    ying_npg_pg("ying_dd_1000")
    reload("reload_1000")
    alm_pqa("alm_pqa_1000")

    tok = time.time()
    print(f"Total benchmark time: {tok - tik}")
