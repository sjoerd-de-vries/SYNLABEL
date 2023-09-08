# A number of standard transformations which can be used with the framework.
import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng()


def plurality_decision(probabilities):
    return np.argmax(probabilities, axis=1)


# Distorted calibration
def random_outcome_sampling(y, fraction):
    new_outcomes = y.copy()

    # Determine the number of outcomes to sample
    num_outcomes = int(fraction * len(y))

    # Randomly select the indices of the outcomes to sample
    sample_indices = np.random.choice(len(y), num_outcomes, replace=False)

    # Sample the outcomes from the prior distribution of the labels
    for i in sample_indices:
        new_outcomes[i] = np.random.choice(y)

    return new_outcomes


def add_gaussian_noise(distribution, noise_size):
    # Create a copy of the data
    noisy_distribution = distribution.copy()

    # Add Gaussian noise to the probabilities in each sample
    for i in range(len(noisy_distribution)):
        probs = noisy_distribution[i, :]
        noisy_probs = probs + np.random.normal(0, noise_size, size=len(probs))
        noisy_probs /= np.sum(noisy_probs)
        noisy_distribution[i, :] = noisy_probs

    return noisy_distribution


# Takes a regular dataset and converts it to have a probabilistic outcome
# Random noise is inserted in proportion to the noise_level and class priors
# Possibly make a variant for which the conversion is done by training
# some clf (LR for instance)
def transform_multiclass_data(y, noise_level=1):
    classes, counts = np.unique(y, return_counts=True)
    labeled_size = len(y)

    priors = counts / labeled_size
    priors = priors.reshape(1, -1)

    noise = rng.uniform(0, noise_level, len(y))
    noise = noise.reshape(-1, 1)

    probs = pd.get_dummies(y).to_numpy("float32")
    inverse_probs = np.abs(probs - 1.0)

    # Calculating the how noise should be applied,
    # while keeping probabilities normalized
    prior_adjusted_noise = noise * priors
    prior_adjusted_noise = prior_adjusted_noise * inverse_probs
    summed_noise = np.sum(prior_adjusted_noise, axis=1)
    noise_correction = probs * summed_noise.reshape(-1, 1)
    prior_adjusted_noise = prior_adjusted_noise - noise_correction

    # Applying the noise to the original labels
    probs = probs + prior_adjusted_noise

    return classes, probs
