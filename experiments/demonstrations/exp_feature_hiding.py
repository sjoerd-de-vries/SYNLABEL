import json
import time
from pathlib import Path

# Generating a Ground Truth dataset
import ground_truth_generation
from joblib import Parallel, delayed

from synlabel.transformations.feature_hiding import sampling_methods

save_path = Path(__file__).parents[2]


D_G = ground_truth_generation.D_G

# Setting up location to store results
start_time = time.time()
save = True
base_save_location = save_path / "experiments/json/feature_hiding"
current_time = str(int(start_time)) + "/"
if save:
    save_location = base_save_location / current_time
    save_location.mkdir(parents=True, exist_ok=True)

# Saving the ground truth
if save:
    g_dict = {}
    g_dict["y"] = D_G.y.tolist()

    # Base metrics
    temp_save_path = save_location / "G_y.json"
    with open(temp_save_path, "w") as outfile:
        json.dump(g_dict, outfile)

# Settings
# change back, samples to 100, repeats 50
repeats = 3
n_jobs = 5
n_features = 17
n_samples = 5


# Transforms the GroundTruth dataset to a PartialGroundTruth dataset via feature hiding
def feature_hiding_exp(feature_index, start_time, repeats):
    features_to_hide = list(range(feature_index + 1))
    resulting_sets = {}

    for iteration in range(repeats):
        for method in sampling_methods:
            result = D_G.to_partial_ground_truth(
                "feature_hiding",
                features_to_hide=features_to_hide,
                samples_per_instance=n_samples,
                sampling_method=method,
            )
            resulting_sets[f"{method}_{iteration}_{feature_index}"] = result.y.tolist()

    return resulting_sets


# Run the experiment in parallel
list_result = Parallel(n_jobs=n_jobs)(
    delayed(feature_hiding_exp)(feature_index, start_time, repeats)
    for feature_index in range(n_features)
)

# Merge the results into a single dictionary
dict_result = {}
for result in list_result:
    dict_result |= result


# Saving the partial ground truth
if save:
    temp_save_path = save_location / "PG_y.json"
    with open(temp_save_path, "w") as outfile:
        json.dump(dict_result, outfile)

current_time = time.time()
print(f"Elapsed time since start: {round(current_time-start_time)}")
