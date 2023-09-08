import math

import numpy as np
import pandas as pd
import scipy
from numpy import linalg as LA

rng = np.random.default_rng()


def one_hot_encoding(y):
    """transforms hard labels into a label distribution

    Parameters
    ----------
    y : array-like
        hard labels

    Returns
    -------
    one_hot
        label distribution
    """

    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1

    return one_hot


class valid_function_generator:
    """Generates a valid function for use with the framwork,
    i.e. a function with the predict() method.

    Parameters
    ----------
    function : classification function
        The function to which a predict() function has to be added.

    Public Methods
    ----------
    predict(X)

    """

    def __init__(self, function):
        self.function = function

    def predict(self, X):
        return np.apply_along_axis(self.function, 1, X)


class decision_function_generator:
    """Generates a valid function for use with the framwork that produces
    hard labels, i.e. a function with the predict() method., from a classifier
    that has the predict_proba method.


    Parameters
    ----------
    clf : classifier
        The classifier to which a decision function has to be applied.
    decision_function : decision function
        The decision function to be applied to clf.

    Public Methods
    ----------
    predict(X)

    """

    def __init__(self, clf, decision_function):
        self.clf = clf
        self.decision_function = decision_function

    def predict(self, X):
        y_distributed = self.clf.predict_proba(X)
        return np.apply_along_axis(self.decision_function, 1, y_distributed)


def in_circle(X):
    """Classification function that classifies points within distance
    1 of the origin as class 0 and as class 1 otherwise

    Parameters
    ----------
    X : array-like
        the datapoint

    Returns
    -------
    label
        the label corresponding to the data point
    """
    # Calculate the distance from the origin (0, 0)
    distance = np.sqrt(X[0] ** 2 + X[1] ** 2)

    # Return 0 if the distance is less than or equal to 1, and 1 otherwise
    if distance <= 1:
        return 0
    else:
        return 1


def mean_total_variation_distance(D_1, D_2):
    """The mean Total Variation Distance between two distributions

    Parameters
    ----------
    D_1 : array-like
        the first distribution
    D_2 : array-like
        the second distribution

    Returns
    -------
    mean_total_var_dist
        the mean Total Variation Distance
    """
    assert D_1.shape[0] == D_2.shape[0]

    n = D_1.shape[0]
    total_var_dist = 0

    for i in range(n):
        var_dist = LA.norm(D_1[i, :] - D_2[i, :], 1)
        total_var_dist += var_dist

    mean_total_var_dist = total_var_dist / (2 * n)

    return mean_total_var_dist


def generate_uniform_noise_matrix(n_classes, flip_prob):
    """This method generates a uniform transition
    matrix, with flip_prob, the probability thath a label is
    transformed away from the diagonal.

    Parameters
    ----------
    n_classes : int
        number of classes in the matrix
    flip_prob : float
        probability of a transition away from the current class

    Returns
    -------
    basis : array-like
        a uniform transition matrix of specified noise level
    """
    basis = np.identity(n_classes)

    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                basis[i, j] -= flip_prob
            else:
                basis[i, j] += flip_prob / (n_classes - 1)

    return basis


def generate_random_transition_matrix(n_classes, flip_prob):
    """This method generates a randomly constructed transition
    matrix, with flip_prob, the probability thath a label is
    transformed away from the diagonal.

    Parameters
    ----------
    n_classes : int
        number of classes in the matrix
    flip_prob : float
        probability of a transition away from the current class

    Returns
    -------
    row_normalized : array-like
        a random normalized transition matrix of specified noise level
    """
    initial_matrix = np.random.uniform(low=0.0, high=1, size=(n_classes, n_classes))
    np.fill_diagonal(initial_matrix, 0)
    row_normalized = initial_matrix / initial_matrix.sum(axis=1, keepdims=1) * flip_prob
    diagonal_value = 1 - flip_prob
    np.fill_diagonal(row_normalized, diagonal_value)

    return row_normalized


def apply_transition_matrix(matrix, dist):
    """Applies a transition matrix to a label distribution

    Parameters
    ----------
    matrix : array-like
        the transition matrix to applu
    dist : array-like
        the distribution to modify

    Returns
    -------
    new_dist : array-like
        the modified label distribution
    """
    new_dist = np.zeros_like(dist)

    for i in range(dist.shape[0]):
        new_dist[i, :] = matrix.dot(dist[i, :])

    return new_dist


def calculate_distance_matrix(data):
    """Calculate the distances between datapoints using LA.norm

    Parameters
    ----------
    data : array-like
        the datapoints

    Returns
    -------
    distance_matric : array-like
        the distances between each two datapoints
    """
    instances = data.shape[0]
    distance_matrix = np.zeros((instances, instances))

    for i in range(instances):
        for j in range(instances):
            if i < j:
                continue
            elif i == j:
                distance_matrix[i, j] = math.inf
            else:
                distance = LA.norm(data[i, :] - data[j, :])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix


def apply_transition_matrix_discrete(labels, matrix):
    """Applies a transition matrix to discrete labels

    Parameters
    ----------
    labels : array-like
        the labels
    matrix : array-like
        the transition matrix to apply

    Returns
    -------
    new_y
        the transformed labels
    """
    new_y = np.zeros(len(labels))

    for label_index in range(len(labels)):
        n_labels = matrix.shape[0]
        prob_dist = matrix[int(labels[label_index]), :]

        new_y[label_index] = rng.choice(n_labels, p=prob_dist)

    return new_y


def apply_transition_matrix_discrete_instance_dependent(
    labels, matrix, sorted_instances
):
    """Apply a transition matrix to existing labels, with the probability
    differing per data point based on their order in sorted_distances.

    Parameters
    ----------
    labels : array-like
        the labels in the dataset
    matrix : array
        The transition matrix which is used for the transformation
    sorted_instances : array-like
        indices ordered such that the first instance has the smallest probability
        of undergoing a transition

    Returns
    -------
    new_y
        the transformed labels
    """
    n_instances = len(sorted_instances)
    n_labels = matrix.shape[0]
    new_y = np.empty(n_instances)

    for instance_index in range(len(sorted_instances)):
        original_index = sorted_instances[instance_index]

        if (instance_index / n_instances) < rng.uniform():
            prob_dist = matrix[int(labels[original_index]), :]

            new_y[original_index] = rng.choice(n_labels, p=prob_dist)
        else:
            new_y[original_index] = labels[original_index]

    return new_y


def sort_dict_by_values(some_dict):
    return dict(sorted(some_dict.items(), key=lambda item: item[1], reverse=True))


def rescale_transition_matrix(transition_matrix, flip_prob):
    """Take a transition matrix with a 1% noise rate and scales it
    to a different noise level.

    Parameters
    ----------
    transition_matrix : array_like
        the original transition matric
    flip_prob : float
        the noise level of the transformed matric

    Returns
    -------
    rescaled_matrix
        the rescaled transition matrix
    """
    # assume 1% corruption rate at start

    classes = transition_matrix.shape[0]
    rescaled_matrix = np.zeros_like(transition_matrix)

    for i in range(classes):
        for j in range(classes):
            if i == j:
                rescaled_matrix[i, j] = 1 - flip_prob
            else:
                rescaled_matrix[i, j] = transition_matrix[i, j] * (flip_prob / 0.01)

    return rescaled_matrix


def average_entropy(array_to_analyse):
    return np.apply_along_axis(scipy.stats.entropy, 1, array_to_analyse).mean()


def average_KL_divergence(d1, d2):
    return np.apply_along_axis(scipy.stats.entropy, 1, (d1, d2)).mean()


def generate_label_index_dict(labels):
    """For each datapoint, collect the indices of data points with the
    same label.

    Parameters
    ----------
    labels : array-like
        the labels in the dataset

    Returns
    -------
    label_index_dict : dict
        contains the indices of points with the same label
    """
    label_index_dict = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_index_dict[label] = []

    for label_index in range(labels.shape[0]):
        label = labels[label_index]
        label_index_dict[label].append(label_index)

    return label_index_dict


def generate_ratio_dict(label_index_dict, distance_matrix, labels):
    """Calculate the ratio of distance to same class to distance to other classes

    Parameters
    ----------
    label_index_dict : dict
        dict containing indices of data points with the same label as 'key'
    distance_matrix : array-like
        a matrix containing the distance between two datapoints
    labels : array-like
        the labels in the dataset

    Returns
    -------
    ratio_dict
        a dictionary containing the ratio's for every datapoint
    """
    ratio_dict = {}

    for key in label_index_dict:
        same_class_labels = label_index_dict[key]
        other_class_labels = [
            i for i in range(labels.shape[0]) if i not in same_class_labels
        ]
        for label_index in same_class_labels:
            min_dist_same = distance_matrix[label_index, same_class_labels].min()
            min_dist_other = distance_matrix[label_index, other_class_labels].min()
            ratio = min_dist_same / min_dist_other
            ratio_dict[label_index] = ratio

    return ratio_dict


def read_data(dataset_name, base_path):
    """Reads in a dataset

    Parameters
    ----------
    dataset_name : string
        filename of the dataset. Has to be .csv
    base_path : path
        directory where the file is located

    Returns
    -------
    dataset: array-like
        the dataset in a numpy array
    """
    dataset = pd.read_csv(base_path / dataset_name, sep=";", header=None)
    dataset = dataset.to_numpy()

    return dataset
