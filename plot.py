from collections import defaultdict
import os
import json
import pickle
from typing import Any
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from utils import (
    hash_string,
    recursive_dict2json,
    dict_permutations,
    update_dict,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "serif",
        # "text.latex.preamble": "\\usepackage{times} ",
        "figure.figsize": (3.25, 2.0086104634371584),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.pad_inches": 0.015,
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "axes.titlesize": 4,
        # --- Tick appearance ---
        # Shorter ticks, thinner ticks
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Tick padding (tight layout)
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        # --- Axes properties ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        # --- Legend style ---
        # "legend.frameon": True,
        # "legend.framealpha": 0.8,
        # "legend.fancybox": False,
        # "legend.edgecolor": "0.2",
    }
)


# mpl.rcParams["xtick.major.size"] = 2
# mpl.rcParams["ytick.major.size"] = 2

# fmt: off
COLORS = [ "r", "b", "g", "y", "orange", "chocolate", "grey", "magenta", "brown", "gold", "teal", "pink", "thistle", "plum", "cyan", "olive"]

ALGO_COLOURS = {
    "NPG-PD": "r",
    "RPG-PD": "r",
    "NPG-RPD": "r",
    "ALM-PQA": "blue",
    "ALM-PPQA": "blue",
    "Muller-NPG-PD": "darkgreen",
    "Ying-NPG-PD": "orange",
    "NPG-DD": "orange",
    "ReLOAD": "pink",
    "PPOLag": "olive",
}

ALGO_LINESTYLES = {
    "NPG-PD": "-",
    "RPG-PD": "-",
    "NPG-RPD": "-",
    "ALM-PQA": "-",
    "ALM-PPQA": "-",
    "Muller-NPG-PD": "-", 
    "Ying-NPG-PD": "-",
    "NPG-DD": "-",
    "ReLOAD": "-",
    "PPOLag": "-",
}

ALGO_LINEWIDTH = {
    "NPG-PD": 0.6, 
    "RPG-PD": 0.6,
    "ALM-PQA": 0.6,
    "ALM-PPQA": 0.6,
    "Muller-NPG-PD": 0.6, 
    "Ying-NPG-PD": 0.6, 
    "NPG-DD": 0.6, 
    "ReLOAD": 0.6, 
    "NPG-RPD": 0.6,
    "PPOLag": 0.6,
}

ALGO_NAME = {
    "ALM-NPG": "NPG-ALM",
    "ALM-PQA": "PQA-ALM",
    "ALM-PPQA": "PPQA-ALM",
    "NPG-PD": "NPG-PD",
    "NPG-RPD": "Lin-RPG-PD",
    "RPG-PD": "RPG-PD",
    "Muller-NPG-PD": "NPG-NRL", 
    "Ying-NPG-PD": "NPG-DD", 
    "NPG-DD": "NPG-DD", 
    "ReLOAD": "ReLOAD", 
    "PPOLag": "PPO-Lag", 
}
# fmt: on


class StatisticTools:
    def __init__(self):
        pass

    def load_source(self, path: str, dump=False) -> None:
        grid_config_dirs = []
        for root, _, files in os.walk(path):
            if "grid_config.json" in files:
                grid_config_dirs.append(files)
                self.grid_config_dir = os.path.join(root, "grid_config.json")
                self.exp_dir = root

        # load the config file of experiment grid
        try:
            with open(self.grid_config_dir, encoding="utf-8") as file:
                self.grid_config = json.load(file)
                if dump:
                    print(json.dumps(self.grid_config, indent=4))

        except FileNotFoundError as error:
            raise FileNotFoundError(
                "The config file is not found in the save directory.",
            ) from error

    def load_variants(self):
        # 1. generate all possible variants of config (excluding seed)
        variants: list[list] = []
        for algo in self.grid_config["algo"]:
            algo_config = deepcopy(self.grid_config["algo"][algo])
            for pinned_config in dict_permutations(algo_config):
                keys = ["env_id", "algo"] + list(pinned_config.keys())
                values = [self.grid_config["env_id"], [algo]] + list(
                    pinned_config.values()
                )
                variants.extend(self._variants(keys, values))

        # 2. for each environment, for each algo, add all variants to list -- process later
        env_id_algo_variant_paths = defaultdict(dict)
        for env_id in self.grid_config["env_id"]:
            for algo in self.grid_config["algo"]:
                env_id_algo_variants = list(
                    filter(
                        lambda variant: (
                            variant["env_id"] == env_id and variant["algo"] == algo
                        ),
                        variants,
                    )
                )
                variant_paths = []
                for variant in env_id_algo_variants:
                    exp_name = (
                        variant["env_id"][:30]
                        + "---"
                        + hash_string(recursive_dict2json(variant))
                    )
                    path = os.path.join(self.exp_dir, exp_name)
                    variant_paths.append((variant, path))
                env_id_algo_variant_paths[env_id][algo] = variant_paths
        return env_id_algo_variant_paths

    def _variants(self, keys: list[str], vals: list[Any]) -> list[dict[str, Any]]:
        """Recursively builds list of valid variants.

        Args:
            keys (list[str]): The keys of the config.
            vals (list[Any]): The values of the config.

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

    def load_paths(self, paths):
        """
        variant_paths: list of paths
        """
        path_pickles = []
        for path in paths:
            pickles = []
            for _, _, files in os.walk(path):
                for f in files:
                    if "output.pickle" in f:
                        with open(os.path.join(path, f), "rb") as handle:
                            dataset = pickle.load(handle)
                        pickles.append(dataset)
            path_pickles.append(pickles)
        return path_pickles

    def best_variants(self, constraint_eps):
        best_variants_paths = defaultdict(list)
        env_id_algo_paths = self.load_variants()
        for env_id in self.grid_config["env_id"]:
            print(f"Env: {env_id}")
            for algo in self.grid_config["algo"]:
                if algo == "PPOLag":
                    constraint_eps = 0.01
                variant_paths = env_id_algo_paths[env_id][algo]
                variants = [variant for variant, _ in variant_paths]
                paths = [path for _, path in variant_paths]
                variants_datasets = self.load_paths(paths)
                assert len(variants_datasets[0]) == len(self.grid_config["seed"])
                best_variant = None
                best_path = None
                best_variant_constraint_violation = None
                smallest_sub_opt_gap = 10000
                for path, variant, datasets in zip(paths, variants, variants_datasets):
                    assert len(datasets) == len(self.grid_config["seed"])

                    # process datasets
                    mean_sub_opt_gap = np.mean(
                        [
                            log_info["v_r_pistar"] - log_info["v_r_pi_T"]
                            for log_info in datasets
                        ]
                    )
                    mean_constraint_violation = np.mean(
                        [log_info["b"] - log_info["v_c_pi_T"] for log_info in datasets]
                    )

                    # when plotting constraint violation, anything negative is good, positive is bad
                    # recall obj: Vc >= b - eps <=> b - Vc <= eps
                    # if b - V^\pi_c <= \eps... we're good
                    if mean_constraint_violation <= constraint_eps and mean_sub_opt_gap:
                        if mean_sub_opt_gap < smallest_sub_opt_gap:
                            smallest_sub_opt_gap = mean_sub_opt_gap
                            best_variant_constraint_violation = (
                                mean_constraint_violation
                            )
                            best_variant = variant
                            best_path = path

                print(f"Algo: {algo}")
                print("Best Parameters", best_variant)
                print("Sub-optimality gap:  ", smallest_sub_opt_gap)
                print("Constraint violation:", best_variant_constraint_violation)
                if best_variant is not None:
                    best_variants_paths[env_id].append((best_variant, best_path))
                else:
                    print(f"No best parameters found for {algo}")
        return best_variants_paths

    def plot_figure_1(self, env_paths, notes: str = ""):
        for env_id, variant_path in env_paths.items():
            if "se" in env_id.lower():
                continue
            fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if "spma" in algo.lower():
                    continue

                if algo == "ALM-NPG":
                    continue

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                # print(f"algo: {algo} env: {env_id}, cv0 = {constraint_violations[0]}")
                # fmt: off
                if algo == 'NPG-PD':
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.6)
                else:
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.8)
                axs[0].set_xlabel("Gradient Evaluations")
                axs[0].set_ylabel(r"Optimality Gap")
                axs[0].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[0].set_ylim(top=0.05, bottom=-0.05)
                axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                axs[0].legend(
                    loc="upper right",
                    bbox_to_anchor=(1.01, 1.01),
                    borderaxespad=0.,
                    fontsize=4.5
                )

                # axs[0].set_yscale("log")

                axs[1].plot(gradient_evaluations, constraint_violations, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[1].set_xlabel("Gradient Evaluations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # toggle 
                axs[1].set_ylim(top=0.1)

                axs[1].legend(
                    loc="upper right",
                    bbox_to_anchor=(1.01, 1.01),
                    borderaxespad=0.,
                    fontsize=4.5
                )

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(0.98, 0.98),
                # )

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if env_id == "ConstrainedDeepSeaTreasure":
                fig.suptitle(f"Constrained Deep Sea Treasure")
            elif env_id == "ConstrainedCliffWorld":
                fig.suptitle(f"Constrained Cliff World")
            else:
                fig.suptitle(f"{env_id}")

            # fig.legend(
            #     lines, labels, loc="lower center", ncol=9, bbox_to_anchor=[0.5, -0.09], fontsize=5
            # )

            plt.subplots_adjust(
                left=0.12,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
                hspace=0.28,
            )

            plt.savefig(
                f"plots/figure_1.png",
                dpi=600,
                bbox_inches="tight",
            )
            print('Saved to: "plots/figure_1.png"')

    def plot_figure_2(self, env_paths, notes: str = ""):
        for env_id, variant_path in env_paths.items():
            fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))

            if "sea" in env_id.lower():
                continue
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if algo == "ALM-NPG":
                    continue

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                # print(f"algo: {algo} env: {env_id}, cv0 = {constraint_violations[0]}")
                # fmt: off
                if algo == 'NPG-PD':
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.6)
                else:
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.8)
                axs[0].set_xlabel("Gradient Evaluations")
                axs[0].set_ylabel(r"Optimality Gap")
                axs[0].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[0].set_ylim(top=0.01, bottom=-0.01)
                axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                # axs[0].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[0].set_yscale("log")

                axs[1].plot(gradient_evaluations, constraint_violations, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[1].set_xlabel("Gradient Evaluations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # toggle 
                axs[1].set_ylim(top=0.1)

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(0.98, 0.98),
                # )

                # axs[0].grid()
                # axs[1].grid()

                # lambdas = log_info["lambas"][: len(gradient_evaluations)]
                # axs[0].plot(np.arange(variant['iterations_outer']+1), lambdas)
                # axs[0].set_xlabel("Iterations")
                # axs[0].set_ylabel(r"Lamdas")
                # fmt: on

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if env_id == "ConstrainedDeepSeaTreasure":
                fig.suptitle(f"Constrained Deep Sea Treasure")
            elif env_id == "ConstrainedCliffWorld":
                fig.suptitle(f"Constrained Cliff World")
            else:
                fig.suptitle(f"{env_id}")

            fig.legend(
                lines,
                labels,
                loc="lower center",
                ncol=9,
                bbox_to_anchor=[0.5, -0.09],
                fontsize=5,
            )

            plt.subplots_adjust(
                left=0.12,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
                hspace=0.28,
            )

            plt.savefig(
                f"plots/figure_2.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('Saved to: "plots/figure_2.png"')

    def plot_figure_3(self, env_paths, notes: str = ""):
        for env_id, variant_path in env_paths.items():
            if "sea" in env_id.lower():
                continue
            fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5))
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                l = ALGO_NAME[algo]

                if l == "NPG-RPD" or l == "RPG-PD":
                    l = "Lin-RPG-PD"

                # print(f"algo: {algo} env: {env_id}, cv0 = {constraint_violations[0]}")
                # fmt: off
                if algo == 'NPG-PD':
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=l, linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.6)
                else:
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=l, linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.8)
                axs[0].set_xlabel("Gradient Evaluations")
                axs[0].set_ylabel(r"Optimality Gap")
                axs[0].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                # axs[0].set_ylim(top=0.01, bottom=-0.01)
                # axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                # axs[0].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[0].set_yscale("log")

                axs[1].plot(gradient_evaluations, constraint_violations, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[1].set_xlabel("Gradient Evaluations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # toggle 
                axs[1].set_ylim(top=0.1)

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(0.98, 0.98),
                # )

                # axs[0].grid()
                # axs[1].grid()

                # lambdas = log_info["lambas"][: len(gradient_evaluations)]
                # axs[0].plot(np.arange(variant['iterations_outer']+1), lambdas)
                # axs[0].set_xlabel("Iterations")
                # axs[0].set_ylabel(r"Lamdas")
                # fmt: on

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if env_id == "ConstrainedDeepSeaTreasure":
                fig.suptitle(f"Constrained Deep Sea Treasure")
            elif env_id == "ConstrainedCliffWorld":
                fig.suptitle(f"Constrained Cliff World")
            else:
                fig.suptitle(f"{env_id}")

            fig.legend(
                lines,
                labels,
                loc="lower center",
                ncol=9,
                bbox_to_anchor=[0.5, -0.09],
                fontsize=5,
            )

            plt.subplots_adjust(
                left=0.12,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
                hspace=0.28,
            )

            plt.savefig(
                f"plots/figure_3.png",
                dpi=300,
                bbox_inches="tight",
            )
            print('Saved to: "plots/figure_3.png"')

    def plot_figure_appendix(self, env_paths, notes: str = ""):
        seen_alm = False
        for env_id, variant_path in env_paths.items():
            fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.0))
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if algo == "NPG-ALM":
                    continue

                if algo == "ALM-NPG":
                    continue

                # if algo == "ALM-PQA" and seen_alm:
                #     continue

                # if "ALM-PQA":
                #     seen_alm = True

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                # print(f"algo: {algo} env: {env_id}, cv0 = {constraint_violations[0]}")
                # fmt: off
                if algo == 'NPG-PD':
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.6)
                else:
                    axs[0].plot(gradient_evaluations, sub_opt_gap, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo], alpha=0.8)
                axs[0].set_xlabel("Gradient Evaluations")
                axs[0].set_ylabel(r"Optimality Gap")
                axs[0].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[0].set_ylim(top=0.01, bottom=-0.01)
                axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                # axs[0].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[0].set_yscale("log")

                axs[1].plot(gradient_evaluations, constraint_violations, label=ALGO_NAME[algo], linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[1].set_xlabel("Gradient Evaluations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], color="black", linestyles="--", linewidth=0.5, alpha=0.6)
                axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # toggle 
                axs[1].set_ylim(top=0.1)

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(1.01, 1.01),
                #     borderaxespad=0.,
                #     fontsize=4.5
                # )

                # axs[1].legend(
                #     loc="upper right",
                #     bbox_to_anchor=(0.98, 0.98),
                # )

                # axs[0].grid()
                # axs[1].grid()

                # lambdas = log_info["lambas"][: len(gradient_evaluations)]
                # axs[0].plot(np.arange(variant['iterations_outer']+1), lambdas)
                # axs[0].set_xlabel("Iterations")
                # axs[0].set_ylabel(r"Lamdas")
                # fmt: on

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if env_id == "ConstrainedDeepSeaTreasure":
                fig.suptitle(f"Constrained Deep Sea Treasure")
            elif env_id == "ConstrainedCliffWorld":
                fig.suptitle(f"Constrained Cliff World")
            else:
                fig.suptitle(f"{env_id}")

            fig.legend(
                lines,
                labels,
                loc="lower center",
                ncol=9,
                bbox_to_anchor=[0.5, -0.09],
                fontsize=5,
            )

            plt.subplots_adjust(
                left=0.12,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
                hspace=0.28,
            )

            plt.savefig(
                f"appendix_plots/{env_id}_best_variants_{notes}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"appendix_plots/{env_id}_best_variants_{notes}.png")

    def plot_figure_appendix_linear(self, env_paths, notes: str = ""):
        for env_id, variant_path in env_paths.items():
            fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.0))
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                l = ALGO_NAME[algo]

                if l == "NPG-RPD" or l == "RPG-PD":
                    l = "Lin-RPG-PD"

                if algo == "NPG-PD":
                    axs[0].plot(
                        gradient_evaluations,
                        sub_opt_gap,
                        label=l,
                        linestyle=ALGO_LINESTYLES[algo],
                        linewidth=ALGO_LINEWIDTH[algo],
                        color=ALGO_COLOURS[algo],
                        alpha=0.6,
                    )
                else:
                    axs[0].plot(
                        gradient_evaluations,
                        sub_opt_gap,
                        label=l,
                        linestyle=ALGO_LINESTYLES[algo],
                        linewidth=ALGO_LINEWIDTH[algo],
                        color=ALGO_COLOURS[algo],
                        alpha=0.8,
                    )
                axs[0].set_xlabel("Gradient Evaluations")
                axs[0].set_ylabel(r"Optimality Gap")
                axs[0].hlines(
                    y=0.0,
                    xmin=0,
                    xmax=gradient_evaluations[-1],
                    color="black",
                    linestyles="--",
                    linewidth=0.5,
                    alpha=0.6,
                )
                # axs[0].set_ylim(top=0.01, bottom=-0.01)
                axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                axs[1].plot(
                    gradient_evaluations,
                    constraint_violations,
                    label=ALGO_NAME[algo],
                    linestyle=ALGO_LINESTYLES[algo],
                    linewidth=ALGO_LINEWIDTH[algo],
                    color=ALGO_COLOURS[algo],
                )
                axs[1].set_xlabel("Gradient Evaluations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(
                    y=0.0,
                    xmin=0,
                    xmax=gradient_evaluations[-1],
                    color="black",
                    linestyles="--",
                    linewidth=0.5,
                    alpha=0.6,
                )
                axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axs[1].set_ylim(top=0.1)

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if env_id == "ConstrainedDeepSeaTreasure":
                fig.suptitle(f"Constrained Deep Sea Treasure")
            elif env_id == "ConstrainedCliffWorld":
                fig.suptitle(f"Constrained Cliff World")
            else:
                fig.suptitle(f"{env_id}")

            fig.legend(
                lines,
                labels,
                loc="lower center",
                ncol=9,
                bbox_to_anchor=[0.5, -0.09],
                fontsize=5,
            )

            plt.subplots_adjust(
                left=0.12,
                right=0.98,
                bottom=0.18,
                top=0.88,
                wspace=0.28,
                hspace=0.28,
            )

            plt.savefig(
                f"appendix_plots/{env_id}_best_variants_{notes}.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(f"appendix_plots/{env_id}_best_variants_{notes}.png")

    def plot_best_variants(self, env_paths, notes: str = ""):
        for env_id, variant_path in env_paths.items():
            fig, axs = plt.subplots(1, 2)
            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                v_r_pistar = log_info["v_r_pistar"]

                algo = variant["algo"]

                if "spma" in algo.lower():
                    continue

                if "iterations_alm" in variant:
                    variant["iterations_outer"] = variant.pop("iterations_alm")

                if "iterations_projected_pqa" in variant:
                    variant["iterations_inner"] = variant.pop(
                        "iterations_projected_pqa"
                    )

                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                v_r_pis = np.array(log_info["v_r_pis"])[: len(gradient_evaluations)]
                sub_opt_gap = v_r_pistar - v_r_pis

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                # print(f"algo: {algo} env: {env_id}, cv0 = {constraint_violations[0]}")
                # fmt: off
                axs[0].plot(gradient_evaluations, sub_opt_gap, label=algo, linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[0].set_xlabel("Iterations")
                axs[0].set_ylabel(r"Optimality Gap")
                # axs[0].set_yscale("log")

                axs[1].plot(gradient_evaluations, constraint_violations, label=algo, linestyle=ALGO_LINESTYLES[algo], linewidth=ALGO_LINEWIDTH[algo], color=ALGO_COLOURS[algo])
                axs[1].set_xlabel("Iterations")
                axs[1].set_ylabel(r"Constraint Violation")
                axs[1].hlines(y=0.0, xmin=0, xmax=gradient_evaluations[-1], label="b", color="black", linestyles="--")
                # toggle 
                axs[1].set_ylim(top=0.1)

                # lambdas = log_info["lambas"][: len(gradient_evaluations)]
                # axs[0].plot(np.arange(variant['iterations_outer']+1), lambdas)
                # axs[0].set_xlabel("Iterations")
                # axs[0].set_ylabel(r"Lamdas")
                # fmt: on

            # Plot last 10% constraint violation data into a subfigure of it's plot
            inset_ax = inset_axes(axs[1], width="45%", height="45%", loc="upper right")

            for variant, path in variant_path:
                log_info = self.load_paths([path])[0][0]

                algo = variant["algo"]
                iterations_outer = variant["iterations_outer"]
                iterations_inner = variant.get("iterations_inner", None)
                gradient_evaluations = self.algo_to_gradient_evaluations(
                    algo, iterations_outer, iterations_inner
                )

                b = log_info["b"]
                v_c_pis = np.array(log_info["v_c_pis"])[: len(gradient_evaluations)]
                constraint_violations = b - v_c_pis

                # Calculate the index for the last 10%
                last_10_percent_idx = int(0.9 * len(gradient_evaluations))
                grad_eval_last_10 = gradient_evaluations[last_10_percent_idx:]
                constraint_violations_last_10 = constraint_violations[
                    last_10_percent_idx:
                ]

                inset_ax.plot(
                    grad_eval_last_10,
                    constraint_violations_last_10,
                    linestyle=ALGO_LINESTYLES[algo],
                    linewidth=ALGO_LINEWIDTH[algo],
                    color=ALGO_COLOURS[algo],
                )

            inset_ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.7)
            # inset_ax.set_xlabel("Gradient Evaluations", fontsize=10)
            # inset_ax.set_ylabel("Constraint Violations", fontsize=10)
            # inset_ax.tick_params(labelsize=6)

            lines, labels = fig.axes[0].get_legend_handles_labels()
            if "40" in notes:
                fig.suptitle(f"{env_id} - d=40")
                if "50" in notes:
                    fig.suptitle(f"{env_id} - d=40, n=50")
            elif "60" in notes:
                fig.suptitle(f"{env_id} - d=60")
                if "50" in notes:
                    fig.suptitle(f"{env_id} - d=60, n=50")
            else:
                fig.suptitle(f"{env_id}")
            fig.legend(
                lines, labels, loc="lower center", ncol=9, bbox_to_anchor=[0.5, -0.08]
            )
            plt.savefig(
                f"det_plots/{env_id}_best_variants_{notes}.png",
                # dpi=200,
                bbox_inches="tight",
            )
            print(f"det_plots/{env_id}_best_variants_{notes}.png")

    def algo_to_gradient_evaluations(
        self, algo, iterations_outer, iterations_inner=None
    ):
        # # to be clean, everything should be of len(gradient_evaluations); I have no idea what I wrote here
        if algo == "OGDA-PD":
            gradient_evaluations = np.arange(0, iterations_outer * 4, 4)
        elif algo in ["NPG-PD", "ReLOAD", "RPG-PD", "NPG-RPD", "PPOLag"]:
            gradient_evaluations = np.arange(0, iterations_outer * 1, 1)
        elif algo in ["test_experiment", "test_experiment_2", "Muller-NPG-PD"]:
            gradient_evaluations = np.arange(0, iterations_outer * 1, 1)
        elif "ALM" in algo or "Ying" or "DD" in algo:
            gradient_evaluations = np.arange(
                iterations_inner,
                iterations_outer * iterations_inner + 1,
                iterations_inner,
            )
            # NOTE: you saved the inital itation, hence this prepend
            gradient_evaluations = np.insert(gradient_evaluations, 0, 0)
        else:
            raise AssertionError(f"Unknown algo: {algo}")
        return gradient_evaluations


def best_versions(paths, notes=""):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(0.01).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_best_variants(best_variant_paths, notes)


def figure_1(paths, notes=""):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(0.001).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_figure_1(best_variant_paths, notes)


def figure_2(paths, notes=""):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(0.001).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_figure_2(best_variant_paths, notes)


def figure_3(paths, notes="", eps=0.01):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(eps).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_figure_3(best_variant_paths, notes)


def figure_appendix(paths, notes="", eps=0.0):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(eps).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_figure_appendix(best_variant_paths, notes)


def figure_appendix_linear(paths, notes="", eps=0.0):
    st = StatisticTools()
    best_variant_paths = defaultdict(list)
    for path in paths:
        st.load_source(path)  # sets self.grid_config
        for env, variantpath in st.best_variants(eps).items():
            best_variant_paths[env].extend(variantpath)
    st.plot_figure_appendix_linear(best_variant_paths, notes)


if __name__ in "__main__":
    figure_1(
        [
            "./runs/npg_pd_1000/",
            "./runs/alm_pqa_1000/",
        ],
        "figure_1",
    )

    figure_2(
        [
            "./runs/ding_rpg_pd_1000/",
            "./runs/alm_pqa_1000",
            "./runs/muller_npg_pd_1000",
            "./runs/ying_dd_1000",
            "./runs/reload_1000",
            "./runs/ppo_lag_one_hot_1000",
        ],
        "figure_2",
    )

    figure_3(
        [
            "./runs/alm_linear_d_60_1000",
            "./runs/npg_pd_linear_d_60_1000",
            "./runs/reload_linear_d_60_1000",
            "./runs/ppo_lag_d_60_1000",
        ],
        "figure_3",
    )

    figure_appendix(
        [
            "./runs/ding_rpg_pd_1000/",
            "./runs/alm_pqa_1000",
            "./runs/muller_npg_pd_1000",
            "./runs/ying_dd_1000",
            "./runs/reload_1000",
            "./runs/ppo_lag_one_hot_1000",
        ],
        "figure_2_appendix_0.001",
        0.001,
    )

    figure_appendix(
        [
            "./runs/ding_rpg_pd_1000/",
            "./runs/alm_pqa_1000",
            "./runs/muller_npg_pd_1000",
            "./runs/ying_dd_1000",
            "./runs/reload_1000",
            "./runs/ppo_lag_one_hot_1000",
        ],
        "figure_2_appendix_0.01",
        0.01,
    )

    figure_appendix_linear(
        [
            "./runs/alm_linear_d_40_1000",
            "./runs/npg_pd_linear_d_40_1000",
            "./runs/reload_linear_d_40_1000",
            "./runs/ppo_lag_d_40_1000",
        ],
        "figure_3_appendix_40_0.001",
        0.001,
    )

    figure_appendix_linear(
        [
            "./runs/alm_linear_d_60_1000",
            "./runs/npg_pd_linear_d_60_1000",
            "./runs/reload_linear_d_60_1000",
            "./runs/ppo_lag_d_60_1000",
        ],
        "figure_3_appendix_60_0.001",
        0.001,
    )

    figure_appendix_linear(
        [
            "./runs/alm_linear_d_80_1000",
            "./runs/npg_pd_linear_d_80_1000",
            "./runs/reload_linear_d_80_1000",
            "./runs/ppo_lag_d_80_1000",
        ],
        "figure_3_appendix_80_0.001",
        0.001,
    )
