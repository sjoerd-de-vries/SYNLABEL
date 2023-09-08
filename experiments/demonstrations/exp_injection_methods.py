import json
import time
from pathlib import Path

import numpy as np
import synlabel.utils.helper_functions as hf
from joblib import Parallel, delayed

save_path = Path(__file__).parents[2]


start_time = time.time()

## Settings
# feature_hiding_method = "marginal_KDE"
feature_hiding_method = "multivariate_imputation_without_y"

if feature_hiding_method == "marginal_KDE":
    fh_name = "marginal_KDE"
elif feature_hiding_method == "multivariate_imputation_without_y":
    fh_name = "MICE"

n_features = 5
save = True
# set 100 100
repeats_1 = 100
repeats_2 = 100

# Setting up location to store results
base_save_location = save_path / f"experiments/json/{fh_name}"
current_time = str(int(start_time))

if save:
    save_location = base_save_location / current_time
    save_location.mkdir(parents=True, exist_ok=True)

# filenames for different result json files
filename_G = "G_y.json"
filename_PG = "PG_y.json"
filename_base_metrics = "base_metrics.json"
filename_transition_matrices = "transition_matrices.json"
filename_exact_metrics = "exact_metrics.json"
filename_sample_metrics = "sample_metrics.json"

## 1. Getting datset, [replace by rdatasets?]
# Run to import label_noise_framework and obtain a GroundTruthDataset
import ground_truth_generation

D_G = ground_truth_generation.D_G

if save:
    g_dict = {}
    g_dict["y"] = D_G.y.tolist()

    # Base metrics
    temp_save_path = save_location / filename_G
    with open(temp_save_path, "w") as outfile:
        json.dump(g_dict, outfile)


## 2. Generating Partial Ground Truth
features_to_hide = list(range(n_features))
print(features_to_hide)

D_PG = D_G.to_partial_ground_truth(
    "feature_hiding",
    features_to_hide=features_to_hide,
    samples_per_instance=100,
    sampling_method=feature_hiding_method,
)

if save:
    pg_dict = {}
    pg_dict["y"] = D_PG.y.tolist()

    temp_save_path = save_location / filename_PG
    with open(temp_save_path, "w") as outfile:
        json.dump(pg_dict, outfile)

D_G_distributed = D_G.to_partial_ground_truth("identity").y

result_dict_base = {}
result_dict_base["var_dist"] = hf.mean_total_variation_distance(D_G_distributed, D_PG.y)
result_dict_base["kl_divergence"] = hf.average_KL_divergence(D_G_distributed, D_PG.y)
result_dict_base["D_G_entropy"] = hf.average_entropy(D_G_distributed)
result_dict_base["D_PG_entropy"] = hf.average_entropy(D_PG.y)

if save:
    # Base metrics
    temp_save_path = save_location / filename_base_metrics
    with open(temp_save_path, "w") as outfile:
        json.dump(result_dict_base, outfile)

## 3. Construct base transition matrices
n_classes = len(np.unique(D_G.y))

uniform_base = hf.generate_uniform_noise_matrix(n_classes, 0.01)
random_base_1 = hf.generate_random_transition_matrix(n_classes, 0.01)
random_base_2 = hf.generate_random_transition_matrix(n_classes, 0.01)
random_base_3 = hf.generate_random_transition_matrix(n_classes, 0.01)

transition_matrices = {}
transition_matrices["uniform"] = uniform_base.tolist()
transition_matrices["random_1"] = random_base_1.tolist()
transition_matrices["random_2"] = random_base_2.tolist()
transition_matrices["random_3"] = random_base_3.tolist()

if save:
    # Transition matrices
    temp_save_path = save_location / filename_transition_matrices
    with open(temp_save_path, "w") as outfile:
        json.dump(transition_matrices, outfile)


def run_noise_exp(corruption_prob):
    result_dict_exact = {}

    print()
    print(f"Current corruption probability: {corruption_prob}")
    print()

    # Construct specific noise matrices
    random_1 = hf.rescale_transition_matrix(random_base_1, corruption_prob)
    random_2 = hf.rescale_transition_matrix(random_base_2, corruption_prob)
    random_3 = hf.rescale_transition_matrix(random_base_3, corruption_prob)
    uniform = hf.rescale_transition_matrix(uniform_base, corruption_prob)

    # 4a. Apply noise injection methods
    D_G_uniform = hf.apply_transition_matrix(uniform, D_G_distributed)
    D_PG_uniform = hf.apply_transition_matrix(uniform, D_PG.y)
    D_G_random_1 = hf.apply_transition_matrix(random_1, D_G_distributed)
    D_PG_random_1 = hf.apply_transition_matrix(random_1, D_PG.y)
    D_G_random_2 = hf.apply_transition_matrix(random_2, D_G_distributed)
    D_PG_random_2 = hf.apply_transition_matrix(random_2, D_PG.y)
    D_G_random_3 = hf.apply_transition_matrix(random_3, D_G_distributed)
    D_PG_random_3 = hf.apply_transition_matrix(random_3, D_PG.y)

    ## 5a. Calculate metrics

    ## D_G to D_G noise
    # uniform
    d1 = D_G_distributed
    d2 = D_G_uniform
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,G_uniform"] = (mtvardist, kldivergence, entropy)

    # random 1
    d2 = D_G_random_1
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,G_random_1"] = (mtvardist, kldivergence, entropy)

    # random 2
    d2 = D_G_random_2
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,G_random_2"] = (mtvardist, kldivergence, entropy)

    # random 3
    d2 = D_G_random_3
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,G_random_3"] = (mtvardist, kldivergence, entropy)

    ## D_PG to D_PG noise
    # uniform
    d1 = D_PG.y
    d2 = D_PG_uniform
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["PG,PG_uniform"] = (mtvardist, kldivergence, entropy)
    # random 1
    d2 = D_PG_random_1
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["PG,PG_random_1"] = (mtvardist, kldivergence, entropy)

    # random 2
    d2 = D_PG_random_2
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["PG,PG_random_2"] = (mtvardist, kldivergence, entropy)

    # random 3
    d2 = D_PG_random_3
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["PG,PG_random_3"] = (mtvardist, kldivergence, entropy)

    ## D_G to D_PG noise
    # uniform
    d1 = D_G_distributed
    d2 = D_PG_uniform
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,PG_uniform"] = (mtvardist, kldivergence, entropy)

    # random 1
    d2 = D_PG_random_1
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,PG_random_1"] = (mtvardist, kldivergence, entropy)

    # random 2
    d2 = D_PG_random_2
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,PG_random_2"] = (mtvardist, kldivergence, entropy)

    # random 3
    d2 = D_PG_random_3
    mtvardist = hf.mean_total_variation_distance(d1, d2)
    kldivergence = hf.average_KL_divergence(d1, d2)
    entropy = hf.average_entropy(d2)
    result_dict_exact["G,PG_random_3"] = (mtvardist, kldivergence, entropy)

    ## 4b. Transform PG to LD no noise

    D_LD = D_PG.to_distributed_observed("identity")

    ## 5b. Transform LD to D by sampling

    result_dict_samples = {}

    # Repeated sampling of LD to obtain D
    for i in range(repeats_1):
        D_D_uniform_distributed = np.zeros_like(D_LD.y)
        D_D_random_1_distributed = np.zeros_like(D_LD.y)
        D_D_random_2_distributed = np.zeros_like(D_LD.y)
        D_D_random_3_distributed = np.zeros_like(D_LD.y)
        D_D_uniform_ID_distributed = np.zeros_like(D_LD.y)
        D_D_random_1_ID_distributed = np.zeros_like(D_LD.y)
        D_D_random_2_ID_distributed = np.zeros_like(D_LD.y)
        D_D_random_3_ID_distributed = np.zeros_like(D_LD.y)

        if i % 5 == 0:
            print(f"Noise rate: {corruption_prob}, iteration: {i}")

        D_D = D_LD.to_discrete_observed("sample")

        # Repeated noise injection
        for j in range(repeats_2):
            # Uniform transition
            D_D_uniform_discrete = hf.apply_transition_matrix_discrete(D_D.y, uniform)
            D_D_uniform_distributed += hf.one_hot_encoding(
                D_D_uniform_discrete.astype(int)
            )

            # Random transition 1
            D_D_random_1_discrete = hf.apply_transition_matrix_discrete(D_D.y, random_1)
            D_D_random_1_distributed += hf.one_hot_encoding(
                D_D_random_1_discrete.astype(int)
            )

            # Random transition 2
            D_D_random_2_discrete = hf.apply_transition_matrix_discrete(D_D.y, random_2)
            D_D_random_2_distributed += hf.one_hot_encoding(
                D_D_random_2_discrete.astype(int)
            )

            # Random transition 3
            D_D_random_3_discrete = hf.apply_transition_matrix_discrete(D_D.y, random_2)
            D_D_random_3_distributed += hf.one_hot_encoding(
                D_D_random_3_discrete.astype(int)
            )

            # Construct ordering based on NN

            # Calculate the distances between each point
            # Do normalise before this.
            distance_matrix = hf.calculate_distance_matrix(D_D.X)

            label_index_dict = hf.generate_label_index_dict(D_D.y)

            # Calculate the ratio of distance to same class to distance to other classes
            ratio_dict = hf.generate_ratio_dict(
                label_index_dict, distance_matrix, D_D.y
            )

            sorted_ratio_dict = hf.sort_dict_by_values(ratio_dict)
            sorted_instances = list(sorted_ratio_dict.keys())

            # Instance dependent uniform transition
            # Make sure noise matrix contains twice the noise
            uniform_times_2 = hf.rescale_transition_matrix(
                uniform_base, corruption_prob * 2
            )
            D_D_uniform_ID_discrete = (
                hf.apply_transition_matrix_discrete_instance_dependent(
                    D_D.y, uniform_times_2, sorted_instances
                )
            )
            D_D_uniform_ID_distributed += hf.one_hot_encoding(
                D_D_uniform_ID_discrete.astype(int)
            )

            # random 1
            random_1_times_2 = hf.rescale_transition_matrix(
                random_base_1, corruption_prob * 2
            )
            D_D_random_1_ID_discrete = (
                hf.apply_transition_matrix_discrete_instance_dependent(
                    D_D.y, random_1_times_2, sorted_instances
                )
            )
            D_D_random_1_ID_distributed += hf.one_hot_encoding(
                D_D_random_1_ID_discrete.astype(int)
            )

            # random 2
            random_2_times_2 = hf.rescale_transition_matrix(
                random_base_2, corruption_prob * 2
            )
            D_D_random_2_ID_discrete = (
                hf.apply_transition_matrix_discrete_instance_dependent(
                    D_D.y, random_2_times_2, sorted_instances
                )
            )
            D_D_random_2_ID_distributed += hf.one_hot_encoding(
                D_D_random_2_ID_discrete.astype(int)
            )

            # random 3
            random_3_times_2 = hf.rescale_transition_matrix(
                random_base_3, corruption_prob * 2
            )
            D_D_random_3_ID_discrete = (
                hf.apply_transition_matrix_discrete_instance_dependent(
                    D_D.y, random_3_times_2, sorted_instances
                )
            )
            D_D_random_3_ID_distributed += hf.one_hot_encoding(
                D_D_random_3_ID_discrete.astype(int)
            )

        ## Measure results
        ## D_PG to D_PG noise
        # uniform
        d1 = D_PG.y
        d2 = D_D_uniform_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_uniform"] = (mtvardist, kldivergence, entropy)

        # random 1
        d2 = D_D_random_1_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_1"] = (mtvardist, kldivergence, entropy)

        # random 2
        d2 = D_D_random_2_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_2"] = (mtvardist, kldivergence, entropy)

        # random 3
        d2 = D_D_random_2_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_3"] = (mtvardist, kldivergence, entropy)

        ## D_G to D_PG noise
        # uniform
        d1 = D_G_distributed
        d2 = D_D_uniform_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_uniform"] = (mtvardist, kldivergence, entropy)

        # random 1
        d2 = D_D_random_1_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_1"] = (mtvardist, kldivergence, entropy)

        # random 2
        d2 = D_D_random_2_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_2"] = (mtvardist, kldivergence, entropy)

        # random 3
        d2 = D_D_random_3_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_3"] = (mtvardist, kldivergence, entropy)

        # Measure results - Instance dependent
        ## D_PG to D_PG noise
        # uniform
        d1 = D_PG.y
        d2 = D_D_uniform_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_uniform_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 1
        d2 = D_D_random_1_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_1_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 2
        d2 = D_D_random_2_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_2_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 3
        d2 = D_D_random_3_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),PG,D_random_3_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        ## D_G to D_PG noise
        # uniform
        d1 = D_G_distributed
        d2 = D_D_uniform_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_uniform_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 1
        d2 = D_D_random_1_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_1_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 2
        d2 = D_D_random_2_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_2_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

        # random 3
        d2 = D_D_random_3_ID_distributed / repeats_2
        mtvardist = hf.mean_total_variation_distance(d1, d2)
        kldivergence = hf.average_KL_divergence(d1, d2)
        entropy = hf.average_entropy(d2)
        result_dict_samples[f"({i}),G,D_random_3_ID"] = (
            mtvardist,
            kldivergence,
            entropy,
        )

    end_time = time.time()
    print(f"Elapsed time: {round(end_time-start_time,2)}")

    return (corruption_prob, result_dict_exact, result_dict_samples)


## Loop over different noise levels
# Hardcoded to prevent rounding errors for file name saving
corruption_probabilities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
list_result = Parallel(n_jobs=9)(
    delayed(run_noise_exp)(noise) for noise in corruption_probabilities
)

if save:
    exact_dict_result = {}
    sample_dict_result = {}

    for result in list_result:
        exact_dict_result[result[0]] = result[1]
        sample_dict_result[result[0]] = result[2]

    # Exact metrics
    temp_save_path = save_location / filename_exact_metrics
    with open(temp_save_path, "w") as outfile:
        json.dump(exact_dict_result, outfile)

    # Sample metrics
    temp_save_path = save_location / filename_sample_metrics
    with open(temp_save_path, "w") as outfile:
        json.dump(sample_dict_result, outfile)
