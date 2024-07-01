import json
from copy import deepcopy

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import synlabel.utils.helper_functions as hf
from synlabel import DiscreteObservedDataset


def get_feature_ranking(coefficients):
    return np.argsort(np.abs(coefficients))[::-1]


def construct_ground_truth(data, clf):
    data = data.copy()
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # shuffle
    shuffled_indices = np.random.permutation(len(y))
    X = X[shuffled_indices].copy()
    y = y[shuffled_indices].copy()

    # Initialize a Discrete Obseved Dataset
    D_OH = DiscreteObservedDataset(X=X, y=y)

    # Fit a deterministic function
    clf.fit(X, y)

    # Construct the Ground Truth dataset
    D_G = D_OH.to_ground_truth("function_based", function=clf)

    return D_G, X, y


def generate_PG(
    D_G,
    n_features_to_hide=1,
    n_samples=100,
    method="multivariate_kde_scipy",
    reverse_feature_order=False,
    model_importance=True,
    verbose=False,
):
    """Generates a D_G and D_PG dataset for testing purposes from another dataset.

    Returns
    -------
    D_G, D_PG : GroundTruthDataset, PartialGroundTruthDataset
    """
    X, y = D_G.X, D_G.y

    if verbose:
        # Measure disagreement between new and old labels
        difference = (y != D_G.y).sum()
        print(f"Difference GT - original: {difference}")
        print(f"% difference: {(100*difference / len(y)).round(2)}")

    if model_importance == True:
        if hasattr(D_G.func, "feature_importances_"):
            # in case of a tree based method
            importances = D_G.func.feature_importances_
        elif hasattr(D_G.func, "coef_"):
            importances = D_G.func.coef_[0]
        else:
            raise ValueError("Model does not have feature importances or coefficients")

    if verbose:
        print(importances)

    # From strongest to weakest correlation
    feature_ranking = get_feature_ranking(importances)

    if verbose:
        print(feature_ranking)

    if reverse_feature_order:
        features_to_hide = feature_ranking[-n_features_to_hide:].astype(int).tolist()
    else:
        features_to_hide = feature_ranking[:n_features_to_hide].astype(int).tolist()

    if verbose:
        print(features_to_hide)

    D_PG = D_G.to_partial_ground_truth(
        "feature_hiding",
        features_to_hide=features_to_hide,
        samples_per_instance=n_samples,
        sampling_method=method,
    )

    return D_PG


def generate_uncertain_sets(
    dataset="keel_vehicle.csv", clf=RandomForestClassifier(), save=False
):

    # Set relevant paths
    data_path = Path.cwd().parents[1] / "data" / dataset
    save_path = Path.cwd().parents[0] / "json"

    # Load the dataset
    dataset = pd.read_csv(data_path, sep=";", header=None)
    dataset = dataset.to_numpy()
    data = dataset

    # Dictionary containing the different sets
    dataset_dict = {}

    # Construct the ground truth dataset
    D_G, X, y = construct_ground_truth(data, clf)

    # Convert the D_G to a distributed dataset
    D_PG_identity = D_G.to_partial_ground_truth("identity")
    dataset_dict["D_PG_identity"] = D_PG_identity

    noise_levels = []

    # MTVD 15, 30, 45
    for n_features, set_name in zip([8, 11, 13], ["D_PG_1", "D_PG_2", "D_PG_3"]):

        # Generate D_PG for different number of features hidden
        D_PG = generate_PG(
            D_G,
            n_features_to_hide=n_features,
            n_samples=1000,
            method="multivariate_kde_sklearn",
            reverse_feature_order=True,
        )
        dataset_dict[set_name] = D_PG
        mtvd = hf.mean_total_variation_distance(D_PG_identity.y, D_PG.y)
        noise_levels.append(mtvd)

    noise_levels = np.round(noise_levels, 2)
    n_classes = len(D_G.classes)

    # Generate the uncertain datasets from noise
    uniform_base = hf.generate_uniform_noise_matrix(n_classes, 0.01)
    random_base = hf.generate_random_transition_matrix(n_classes, 0.01)

    transition_matrices = {}
    transition_matrices["uniform"] = uniform_base.tolist()
    transition_matrices["random"] = random_base.tolist()

    if save:
        filename_transition_matrices = "transition_matrices_uncertainty.json"

        # Transition matrices
        temp_save_path = save_path / filename_transition_matrices
        with open(temp_save_path, "w") as outfile:
            json.dump(transition_matrices, outfile)

    print(f"noise levels: {noise_levels}")

    for noise_level, set_name in zip(noise_levels, ["1", "2", "3"]):

        # Construct specific noise matrices
        uniform = hf.rescale_transition_matrix(uniform_base, noise_level)
        random = hf.rescale_transition_matrix(random_base, noise_level)

        # 4a. Apply noise injection methods
        D_PG_y_uniform = hf.apply_transition_matrix(D_PG_identity.y, uniform)
        D_PG_y_random = hf.apply_transition_matrix(D_PG_identity.y, random)
        D_PG_uniform = deepcopy(D_PG_identity)
        D_PG_uniform.y = D_PG_y_uniform
        D_PG_random = deepcopy(D_PG_identity)
        D_PG_random.y = D_PG_y_random

        dataset_dict[f"D_PG_uniform_{set_name}"] = D_PG_uniform
        dataset_dict[f"D_PG_random_{set_name}"] = D_PG_random

    ## 5a. Calculate metrics
    ## D_G to D_G noise
    mtvd_dict = {}
    d1 = D_PG_identity.y
    for set_name, new_set in dataset_dict.items():
        d2 = new_set.y
        mtvd = hf.mean_total_variation_distance(d1, d2)
        mtvd_dict[f"MTVD_{set_name}"] = mtvd

    feature_ranking = get_feature_ranking(D_G.func.feature_importances_)

    return dataset_dict, mtvd_dict, feature_ranking, noise_levels
