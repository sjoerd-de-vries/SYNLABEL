import json
import math
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

save_path = Path(__file__).parent.parent.parent

# Scenarios - Pick one: a combination of method and measure
scenario = 2
if scenario == 1:
    method = "marginal_KDE"
    measure = "var_dist"
elif scenario == 2:
    method = "MICE"
    measure = "entropy"
else:
    raise Exception(
        f"Scenario {scenario} not implemented. Manually add if another combination of method and measure is of interest"
    )

# Settings - should be the same values as in the experiments
repeats_1 = 100  # 100
float_noise_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

# Setting paths
base_path = save_path / "experiments"
if method == "marginal_KDE":
    absolute_result_path = base_path / "json/marginal_KDE/<set_correct_path>"
elif method == "MICE":
    absolute_result_path = base_path / "json/MICE/<set_correct_path>"
save_path = base_path / "figures"

# For loading from json
noise_rates = [str(x) for x in float_noise_rates]

# Loading the results files
# Base metrics
with open(absolute_result_path / "base_metrics.json", "r") as f:
    base_metrics = json.load(f)

# Exact metrics
with open(absolute_result_path / "exact_metrics.json", "r") as f:
    exact_metrics = json.load(f)

# Sample metrics
with open(absolute_result_path / "sample_metrics.json", "r") as f:
    sample_metrics = json.load(f)

# Dictionaries to save metrics
exact_plot_dict = {}
sample_plot_dict = {}
sample_std_plot_dict = {}

# Managing the dictionary keys and setting baseline noise after feature hiding
if measure == "var_dist":
    measure_key = 0
    baseline = base_metrics[measure]
# Will be infinity if supports are not shared, as division by 0 will occur.
elif measure == "kl_divergence":
    measure_key = 1
    baseline = base_metrics[measure]
elif measure == "entropy":
    measure_key = 2
    baseline = base_metrics["D_PG_entropy"]

# Setting noise rate == 0 starting values
for key in exact_metrics["0.05"].keys():
    stripped_key = key[:4]

    if stripped_key == "G,PG":
        start_value = baseline
    else:
        start_value = 0.0

    exact_plot_dict[key] = [start_value]
    for noise_rate in noise_rates:
        exact_plot_dict[key].append(exact_metrics[noise_rate][key][measure_key])


# Generating additive noises for the uniform noise matrix
if measure != "entropy":
    exact_plot_dict["PG,PG_uniform_added"] = [
        i + baseline for i in exact_plot_dict["PG,PG_uniform"]
    ]
exact_plot_dict["G,G_uniform_added"] = [
    i + baseline for i in exact_plot_dict["G,G_uniform"]
]
# The same for the randomly generated noise matrices
for i in range(1, 4):
    if measure != "entropy":
        exact_plot_dict[f"PG,PG_random_{i}_added"] = (
            np.array(exact_plot_dict[f"PG,PG_random_{i}"]) + baseline
        )
    exact_plot_dict[f"G,G_random_{i}_added"] = (
        np.array(exact_plot_dict[f"G,G_random_{i}"]) + baseline
    )

# Calculating the metrics to plot
for key in list(sample_metrics["0.05"].keys())[:16]:
    stripped_key = key[4:]

    if (stripped_key[0] == "G") & (measure != "entropy"):
        start_value = baseline
    else:
        start_value = 0.0

    sample_plot_dict[stripped_key] = [start_value]
    sample_std_plot_dict[stripped_key] = [0.0]

    for noise_rate in noise_rates:
        current_metric_list = []

        for i in range(repeats_1):
            current_metric_list.append(
                sample_metrics[noise_rate][f"({i}),{stripped_key}"][measure_key]
            )

        mean = np.mean(current_metric_list)
        std = np.std(current_metric_list)

        sample_plot_dict[stripped_key].append(mean)
        sample_std_plot_dict[stripped_key].append(std)


# Appending noise rate = 0
plot_noise_rates = np.array([0.0] + float_noise_rates) * 100
baseline_list = [baseline for _ in plot_noise_rates]


def plot_mean_measure(
    measure_dict,
    sample_dict,
    std_dict,
    baseline,
    key_to_label,
    default_cycler,
    measure,
    figsize=(5, 3),
    noise_type="random_1",
    save_dir=save_path,
    label_weight=600,
    label_size=9,
    font_size=6,
    font_weight=600,
    z=1.96,
    n=repeats_1,
    plot_ci=False,
):
    filename = f"{measure}_{noise_type}_{method}.png"

    plt.figure(figsize=figsize)
    plt.rc("axes", prop_cycle=default_cycler)
    plt.rcParams["font.size"] = font_size
    plt.rcParams["font.weight"] = font_weight

    # Plot baseline
    plt.plot(
        plot_noise_rates, baseline, label=r"$\Delta 1$: $D^{PG}$", linestyle="dashed"
    )

    # Read different results from dict depending on what to plot
    if measure == "entropy":
        for key, value in measure_dict.items():
            if (noise_type in key) & ("PG,PG" not in key):
                print(key)
                if key == "G,G_random_1":
                    plt.plot(
                        plot_noise_rates,
                        value,
                        label=key_to_label[key],
                        linestyle="dashed",
                    )
                else:
                    plt.plot(plot_noise_rates, value, label=key_to_label[key])

        for key, value in sample_dict.items():
            if (noise_type in key) & ("PG" not in key):
                plt.plot(
                    plot_noise_rates, value, label=key_to_label[key], linestyle="dotted"
                )

                if plot_ci:
                    min_val = value - z * np.array(std_dict[key]) / math.sqrt(n)
                    max_val = value + z * np.array(std_dict[key]) / math.sqrt(n)
                    plt.fill_between(plot_noise_rates, min_val, max_val, alpha=0.2)

    elif measure == "var_dist":
        for key, value in measure_dict.items():
            if noise_type in key:
                if "added" not in key and "G,PG_uniform" != key:
                    plt.plot(
                        plot_noise_rates,
                        value,
                        label=key_to_label[key],
                        linestyle="dashed",
                    )
                else:
                    plt.plot(plot_noise_rates, value, label=key_to_label[key])

        for key, value in sample_dict.items():
            if (noise_type in key) & ("PG" not in key):
                plt.plot(
                    plot_noise_rates, value, label=key_to_label[key], linestyle="dotted"
                )

                if plot_ci:
                    min_val = value - z * np.array(std_dict[key]) / math.sqrt(n)
                    max_val = value + z * np.array(std_dict[key]) / math.sqrt(n)
                    plt.fill_between(plot_noise_rates, min_val, max_val, alpha=0.2)

    if measure == "var_dist":
        y_label = "Mean Total Variation Distance"
    elif measure == "entropy":
        y_label = "Mean Entropy"

    plt.legend()
    plt.xlabel(
        "Noise Rate (%)",
        fontname="Times New Roman",
        fontsize=label_size,
        weight=label_weight,
    )
    plt.ylabel(
        y_label,
        fontname="Times New Roman",
        fontsize=label_size,
        weight=label_weight,
    )
    plt.savefig(save_dir / filename, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()


if scenario == 1:
    # Setting the correct labels for in the legend
    key_to_label = {}
    key_to_label["G,G_uniform"] = r"$\Delta 2$: $T_r(D^G)$"
    key_to_label["G,PG_uniform"] = r"$T_r(FH(D^{G}))$"
    key_to_label["PG,PG_uniform"] = r"$\Delta 3$: $T_r(D^{PG})$"
    key_to_label["PG,PG_uniform_added"] = r"$\Delta 1 + \Delta 2$"
    key_to_label["G,G_uniform_added"] = r"$\Delta 1 + \Delta 3$"
    key_to_label["G,D_uniform"] = r"$T_r(D^{OH})$"
    key_to_label["G,D_uniform_ID"] = r"$ID(T_r(D^{OH})$"
    default_cycler = cycler(
        color=[
            "#000000",
            "#006583",
            "#71acbb",
            "#000000",
            "#006583",
            "#71acbb",
            "#9abfa1",
            "#3a6a37",
        ]
    )

    plot_mean_measure(
        exact_plot_dict,
        sample_plot_dict,
        sample_std_plot_dict,
        baseline_list,
        key_to_label,
        default_cycler,
        measure,
        noise_type="uniform",
    )
elif scenario == 2:
    # Setting the correct labels for in the legend
    key_to_label = {}
    key_to_label["G,G_random_1"] = r"$\Delta 2$: $T_r(D^G)$"
    key_to_label["G,PG_random_1"] = r"$T_r(FH(D^{G}))$"
    key_to_label["G,G_random_1_added"] = r"$\Delta 1 + \Delta 2$"
    key_to_label["G,D_random_1"] = r"$T_r(D^{OH})$"
    key_to_label["G,D_random_1_ID"] = r"$ID(T_r(D^{OH})$"
    default_cycler = cycler(
        color=["#000000", "#006583", "#000000", "#006583", "#9abfa1", "#3a6a37"]
    )

    plot_mean_measure(
        exact_plot_dict,
        sample_plot_dict,
        sample_std_plot_dict,
        baseline_list,
        key_to_label,
        default_cycler,
        measure,
        noise_type="random_1",
    )
