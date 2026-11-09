import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from typing import Any

from rich.console import Console

from utils import hash_string, recursive_dict2json, update_dict


class ExperimentGrid:
    def __init__(self, experiment_name: str, config: dict[str, Any]) -> None:
        self.experiment_name = experiment_name
        self.config = config
        self._console: Console = Console()

    def print(self):
        self._console.print(
            f"Grid Search \[{self.experiment_name}] over the following parameters",
            style="green bold",
        )
        total_algo_variants = 0
        for key, value in self.config.items():
            self._console.print("", key.ljust(40), style="cyan bold")

            for _, val in enumerate(value):
                print("\t" + json.dumps(val, indent=4, sort_keys=True))
                if isinstance(self.config[key], dict):
                    algo_variants = 1
                    for k, v in self.config[key][val].items():
                        self._console.print(f"\t [yellow bold]{k}[/yellow bold]: {v}")
                        algo_variants *= len(v)
                    total_algo_variants += algo_variants

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_seedless = len(self.config["env_id"]) * total_algo_variants
        print(" Variants, not counting seeds: ".ljust(40), nvars_seedless)

    def save_grid_config(self) -> None:
        """Save experiment grid configurations as json."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "grid_config.json")
        self._console.print(
            "Save with config of experiment grid in grid_config.json",
            style="yellow bold",
        )
        json_config = json.dumps(self.config, indent=4)
        with open(path, encoding="utf-8", mode="w") as f:
            f.write(json_config)

    def _variants(self, keys: list[str], vals: list[Any]) -> list[dict[str, Any]]:
        """Recursively builds list of valid variants.

        Args:
            keys (keys: list[str]): List of keys.
            vals (list[Any]): List of values.

        Returns:
            List of valid variants.
        """
        if len(keys) == 1:
            pre_variants: list[dict[str, Any]] = [{}]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                current_variants = deepcopy(pre_v)
                v_temp = {}
                key_list = keys[0].split(":")
                v_temp[key_list[-1]] = val
                for key in reversed(key_list[:-1]):
                    v_temp = {key: v_temp}
                update_dict(current_variants, v_temp)
                variants.append(current_variants)

        return variants

    def variants(self):
        flat_variants = []
        for algo in self.config["algo"]:
            keys = ["env_id", "algo"] + list(self.config["algo"][algo].keys())
            values = [self.config["env_id"], [algo]] + list(
                self.config["algo"][algo].values()
            )

            if "seed" in self.config.keys():
                keys += ["seed"]
                values.append(self.config["seed"])
            flat_variants.extend(self._variants(keys, values))

        def check_duplicate(var: dict[str, Any]) -> dict[str, Any]:
            """Build the full nested dict version of var, based on key names."""
            new_var: dict[str, Any] = {}
            unflatten_set: set = set()

            for key, value in var.items():
                assert key not in new_var, (
                    "You can't assign multiple values to the same key."
                )
                new_var[key] = value

            # make sure to fill out the nested dict.
            for key in unflatten_set:
                new_var[key] = check_duplicate(new_var[key])

            return new_var

        return [check_duplicate(var) for var in flat_variants]

    def run(self, thunk, num_pool=1, verbose=False):
        self.log_dir = os.path.join("./", "runs", self.experiment_name)

        try:
            assert not os.path.exists(self.log_dir)
        except AssertionError:
            self._console.print(
                f"ERROR: log_dir {self.log_dir} already exists!", style="bold red"
            )
        self.save_grid_config()
        self.print()

        # make the list of all variants.
        variants = self.variants()

        results = []  # holds True / False value if done
        exp_names = []

        with Pool(max_workers=num_pool, mp_context=mp.get_context("spawn")) as executor:
            for idx, var in enumerate(variants):
                print("variant: ", var)

                # {'algo': 'OPG-PD', 'env_id': 'ConstrainedDeepSeaTreasure', 'eta_dual': 0.1, 'eta_primal': 0.1, 'iterations_outer': 10000, 'seed': 2}]
                name_var = deepcopy(var)
                seed = name_var.pop("seed", 0)
                exp_name = recursive_dict2json(name_var)
                hashed_exp_name = (
                    name_var["env_id"][:30] + "---" + hash_string(exp_name)
                )
                exp_names.append(":".join((hashed_exp_name[:5], exp_name)))
                exp_log_dir = os.path.join(self.log_dir, hashed_exp_name, "")

                # save entire variant config to file in log_dir
                self.save_same_exps_config(exp_log_dir, var)

                # pop all non-algorithm specifc parameters from the variant
                algo = var.pop("algo")
                env_id = var.pop("env_id")
                seed = var.pop("seed", 0)

                results.append(
                    executor.submit(
                        thunk,
                        experiment_id=idx,
                        log_dir=exp_log_dir,
                        algo=algo,
                        env_id=env_id,
                        algo_params=var,
                        seed=seed,
                        verbose=verbose,
                    )
                )

        self.save_results(exp_names, variants, results)

    def save_results(
        self,
        exp_names: list[str],
        variants: list[dict[str, Any]],
        results: list,
    ) -> None:
        """Save results to a file.

        Args:
            exp_names (list of str): List of experiment names.
            variants (list[dict[str, Any]]): List of experiment variants.
            results (list): List of experiment results.
        """
        path = os.path.join(self.log_dir, "results.txt")
        str_len = max(len(exp_name) for exp_name in exp_names)
        exp_names = [exp_name.ljust(str_len) for exp_name in exp_names]
        with open(path, "a+", encoding="utf-8") as f:
            for idx, _ in enumerate(variants):
                if is_experiment_done := results[idx].result():
                    f.write(exp_names[idx] + ": ")
                    f.write("finished:" + str(is_experiment_done) + ",")
                    f.write("\n")

    def save_same_exps_config(self, exps_log_dir: str, variant: dict[str, Any]) -> None:
        """Save experiment grid configurations as json.

        Args:
            exps_log_dir (str): Experiment log directory.
            variant (dict[str, Any]): Experiment variant.
        """
        os.makedirs(exps_log_dir, exist_ok=True)
        path = os.path.join(exps_log_dir, "exps_config.json")
        json_config = json.dumps(variant, indent=4)
        with open(path, encoding="utf-8", mode="a+") as f:
            f.write("\n" + json_config)
