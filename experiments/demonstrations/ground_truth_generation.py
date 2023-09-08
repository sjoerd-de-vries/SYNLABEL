from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import synlabel.utils.helper_functions as hf
from synlabel import DiscreteObservedDataset, GroundTruthDataset

dir_path = Path(__file__).parents[2]

rng = np.random.default_rng()

# Settings
verbose = False

## Example 1: how to generate a simulated Ground Truth dataset

# Sample random values from the range [-2, 2] and assign them to X
n = (100, 2)
min = -2
max = 2
X = rng.uniform(min, max, n)

# Set f_G to the in_circle() function
# This means any datapoint within distance 1 of the origin
# is class 0, and any point further away is class 1.
f_G = hf.valid_function_generator(hf.in_circle)
y_G = f_G.predict(X)
X_G = X

D_D = DiscreteObservedDataset(X=X_G, y=y_G, func=f_G)

# Testing
D_G_1 = D_D.to_ground_truth("function_based")
D_G_2 = D_D.to_ground_truth("function_based", function=f_G)

D_D = DiscreteObservedDataset(X=X_G, y=y_G)
# Testing
# D_G_1 = D_D.to_ground_truth('function_based') correctly fails
D_G_2 = D_D.to_ground_truth("function_based", function=f_G)

# Print some results
if verbose:
    print(X_G[:5, :])
    print(y_G[:5])

# Construct the Ground Truth dataset
D_G = GroundTruthDataset(X=X_G, y=y_G, func=f_G)

## Example 2: how to generate a Ground Truth set based on a real dataset.
## This set is used in the demonstrations in the paper as well.

# Obtain a real dataset
data_path = dir_path / "data"
data = hf.read_data("keel_vehicle.csv", data_path)
X = data[:, :-1]
y = data[:, -1].astype(int)

# shuffle
shuffled_indices = np.random.permutation(len(y))
X = X[shuffled_indices].copy()
y = y[shuffled_indices].copy()

# Fit a deterministic function
clf = RandomForestClassifier()
clf.fit(X, y)

# # Set f_G to the clf function
f_D = clf

# Construct the Ground Truth dataset
f_G = f_D
y_G = f_G.predict(X)
X_G = X
D_G_0 = GroundTruthDataset(X=X_G, y=y_G, func=f_G)

# More paths to constructing D_G
D_D = DiscreteObservedDataset(X=X, y=y, func=f_D)
D_G_1 = D_D.to_ground_truth("function_based")
D_G_2 = D_D.to_ground_truth("function_based", function=f_D)
D_PG_1 = D_D.to_partial_ground_truth("function_based")
D_PG_2 = D_D.to_partial_ground_truth("function_based", function=f_D)
D_G_3 = D_PG_2.to_ground_truth("determinstic_decision")

# Even more
D_D = DiscreteObservedDataset(X=X, y=y)
D_G_4 = D_D.to_ground_truth("function_based", function=f_D)
D_PG = D_D.to_partial_ground_truth("function_based", function=f_D)
D_G_5 = D_PG.to_ground_truth("determinstic_decision")

# Finally
clf = RandomForestClassifier()
D_D.learn_function(clf)
D_G = D_D.to_ground_truth("function_based")

# Print some results
if verbose:
    print(X_G[:5, :])
    print(y_G[:5])

if verbose:
    print((D_G.y == D_G_0.y).all())
    print((D_G.y == D_G_1.y).all())
    print((D_G.y == D_G_2.y).all())
    print((D_G.y == D_G_3.y).all())
    print((D_G.y == D_G_4.y).all())
    print((D_G.y == D_G_5.y).all())
