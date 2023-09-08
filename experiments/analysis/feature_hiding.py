import json
import math
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import scipy
from cycler import cycler
from sklearn.metrics import accuracy_score

mod_path = Path(__file__).parents[2]

from synlabel.transformations.feature_hiding import sampling_methods

# Settings - should be the same values as in the experiments
repeats = 50  # 50
n_features_hidden = 17

# Setting paths
base_path = mod_path / "experiments"
load_path = base_path / "json/feature_hiding/<set_correct_path>"
save_path = base_path / "figures/feature_hiding"


# Metrics to calculate
def calculate_summed_entropy(array_to_analyse):
    return np.apply_along_axis(scipy.stats.entropy, 1, array_to_analyse).sum()


def plurality_decision(probabilities):
    return np.argmax(probabilities, axis=1)


# Dictionaries to save metrics
entropy_dict = {}
accuracy_dict = {}
entropy_std_dict = {}
accuracy_std_dict = {}

# Not all methods shown in plot for clarity
sampling_methods.remove("marginal_histogram")
sampling_methods.remove("multivariate_kde_scipy")

# Loading the results files
with open(load_path / "PG_y.json", "r") as f:
    resulting_sets = json.load(f)

with open(load_path / "G_y.json", "r") as f:
    set_gt = json.load(f)["y"]

# Preparing the plotting dictionaries
for method in sampling_methods:
    entropy_dict[method] = []
    accuracy_dict[method] = []
    entropy_std_dict[method] = []
    accuracy_std_dict[method] = []

# Filling the dictionaries with results of interest
for feature_index in range(n_features_hidden):
    features_to_hide = list(range(feature_index + 1))

    for method in sampling_methods:
        average_entropy = []
        plurality_accuracy = []
        entropy_std = []
        plurality_accuracy_std = []

        for iteration in range(repeats):
            new_key = f"{method}_{iteration}_{feature_index}"

            summed_entropy = calculate_summed_entropy(resulting_sets[new_key])
            average_entropy.append(np.array(summed_entropy) / len(set_gt))
            discrete_labels = plurality_decision(resulting_sets[new_key])
            plurality_accuracy.append(accuracy_score(discrete_labels, set_gt))

        mean_average_entropy = np.mean(average_entropy)
        mean_plurality_accuracy = np.mean(plurality_accuracy)
        entropy_dict[method].append(mean_average_entropy)
        accuracy_dict[method].append(mean_plurality_accuracy)
        entropy_std_dict[method].append(np.std(average_entropy))
        accuracy_std_dict[method].append(np.std(plurality_accuracy))

# Setting the correct labels for in the legend
label_dict = {}
label_dict["uniform_independent"] = "uniform"
label_dict["marginal_KDE"] = "marginal KDE"
label_dict["multivariate_kde_sklearn"] = "multivariate KDE"
label_dict["multivariate_imputation_without_y"] = "MICE without y"
label_dict["multivariate_imputation_with_y"] = "MICE with y"


# Plotting the results from exp_feature_hiding.py
def plot_average_entropies(
    result_dict,
    filename,
    figsize=(4, 2.4),
    save_dir=save_path,
    std_dict={},
    plot_ci=False,
    label_weight=600,
    label_size=7.5,
):
    # Differnt line colors
    default_cycler = cycler(
        color=["#001b00", "#3a6a37", "#9abfa1", "#71acbb", "#006583"]
    )

    # Setting up the figure
    plt.figure(figsize=figsize)
    plt.rc("axes", prop_cycle=default_cycler)
    plt.rcParams["font.size"] = 5.5
    plt.rcParams["font.weight"] = 600

    # Iterating over the result dict
    for key, value in result_dict.items():
        plt.plot(range(1, len(value) + 1), value, label=label_dict[key], linewidth=1)

        # For plotting Confidence Intervals
        if plot_ci:
            z = 1.96
            std = np.array(std_dict[key])
            ci = z * std / math.sqrt(repeats)
            min_ci = np.array(value) - ci
            max_ci = np.array(value) + ci
            plt.fill_between(range(1, len(value) + 1), min_ci, max_ci, alpha=0.2)

    font = font_manager.FontProperties(
        family="Times New Roman", style="normal", size=6.5, weight=600
    )
    plt.legend(prop=font)

    # Switching labels based on the filename.
    # Make sure the correct filename is used with the correct result dictionary.
    if "entropy" in filename:
        plt.ylabel(
            "Mean Entropy",
            fontname="Times New Roman",
            fontsize=label_size,
            weight=label_weight,
        )
    elif "accuracy" in filename:
        plt.ylabel(
            "Average Accuracy",
            fontname="Times New Roman",
            fontsize=label_size,
            weight=label_weight,
        )

    plt.xlabel(
        "Features Hidden",
        fontname="Times New Roman",
        fontsize=label_size,
        weight=label_weight,
    )
    plt.savefig(save_dir / filename, dpi=600, transparent=True, bbox_inches="tight")
    plt.show()


# Generating the figures
plot_average_entropies(entropy_dict, filename="average_entropy_features.png")
plot_average_entropies(accuracy_dict, filename="average_accuracy_features.png")
