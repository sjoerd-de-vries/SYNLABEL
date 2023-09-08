import numpy as np

from ..transformations.feature_hiding import Sampler
from ..utils.validation import validate_features_to_hide
from . import discrete_observed as D_dataset
from . import distributed_observed as LD_dataset
from . import partial_ground_truth as PG_dataset
from .abstract_datasets import DiscreteDataset


class GroundTruthDataset(DiscreteDataset):
    """The Ground Truth (G) class. Inherits from DiscreteDataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features).
        The input variables available to the classification task.
    X_complement : array-like of shape (n_samples, n_features).
        The input variables not available to the classification task.
    y : array-like of shape (n_samples, 1).
        The target variable. Hard labels.
    func : object
        An object (function) implementing the "predict" or "predict_proba" method.

    Public Methods
    ----------
    to_partial_ground_truth(specific_method, **kwargs)
    to_distributed_observed(specific_method, **kwargs)
    to_discrete_observed(specific_method, **kwargs)


    See specific documentation of the methods for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _feature_hiding(
        self, to_hide, sampling_method, samples_per_instance, var_types, custom_sampler
    ):
        # sampling method refers to the method, implemented in
        # transformations.feature_hidings, by which the complement data
        # is sampled to construct the posterior from via f_PG

        if validate_features_to_hide(self.X, to_hide):
            # Dividing X
            to_keep = [i for i in range(self.X.shape[1]) if i not in to_hide]
            X_PG = self.X[:, to_keep]
            X_complement_PG = self.X[:, to_hide]

            sampler = Sampler(sampling_method, X_PG, X_complement_PG, self.y, var_types)
            X_complement_samples = sampler.sample(samples_per_instance, custom_sampler)

            # Duplicate X in original order here, which includes future X'
            # then replace X' columns
            X_extended = np.tile(self.X, (samples_per_instance, 1))
            for column_index in range(len(to_hide)):
                X_extended[:, to_hide[column_index]] = X_complement_samples[
                    :, column_index
                ]

            # Generate predictions
            y_extended = self.func.predict(X_extended)

            new_y_shape = (self.y.shape[0], len(np.unique(self.y)))
            y_PG = np.zeros(new_y_shape)

            for i in range(y_extended.shape[0]):
                modulated_index = i % self.y.shape[0]
                new_label = y_extended[i]
                y_PG[modulated_index, new_label] += 1 / samples_per_instance

            func_PG = self.func
        else:
            raise Exception("Incorrect list of features to hide provided")

        return X_PG, X_complement_PG, y_PG, func_PG

    def to_partial_ground_truth(self, specific_method, **kwargs):
        """Transforms the current dataset into a Partial Ground Truth dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {feature_hiding, identity}
        **kwargs : dict
            features_to_hide (specific_method=feature_hiding):
                The indices of the features to hide.
            sampling_method (specific_method=feature_hiding):
                The sampling method used. options: {'uniform_independent',
                'marginal_histogram', 'marginal_KDE', 'multivariate_kde_scipy',
                'multivariate_kde_sklearn','multivariate_imputation_without_y',
                'multivariate_imputation_with_y', 'custom'}
            samples_per_instance (specific_method=feature_hiding):
                Number of samples taken per instance in the set.
            var_types (specific_method=feature_hiding):
                The types of the different variables. Has to be a string, containing:
                ['c': Continuous 'u': Unordered (Discrete) 'o': Ordered (Discrete)].
                Has to be of the same length as the total number of features.
                Example: 'ccccucoo'
            custom_sampler (specific_method=feature_hiding)
                A custom sampling function.
                Has to return sampled data of size
                (X_complement.shape[1], X_complement.shape[0] * samples_per_instance)

        Returns
        -------
        PartialGroundTruthDataset
            a newly constructed Partial Ground Truth dataset
        """
        # Call specific transformation
        if specific_method == "feature_hiding":
            features_to_hide = kwargs["features_to_hide"]
            sampling_method = kwargs["sampling_method"]
            samples_per_instance = kwargs["samples_per_instance"]
            var_types = kwargs.get("var_types", [])
            custom_sampler = kwargs.get("custom_sampler", None)
            X, X_complement, y, func = self._feature_hiding(
                features_to_hide,
                sampling_method,
                samples_per_instance,
                var_types,
                custom_sampler,
            )
        elif specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("distributed")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = PG_dataset.PartialGroundTruthDataset(
            X=X, X_complement=X_complement, y=y, func=func
        )

        return transformed_dataset

    def to_distributed_observed(self, specific_method, **kwargs):
        """Transforms the current dataset into a Label Distribution dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {identity}
        **kwargs : dict
            not in use

        Returns
        -------
        DistributedObservedDataset
            a newly constructed Label Distribution dataset
        """

        # Call specific transformation
        if specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("distributed")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = LD_dataset.DistributedObservedDataset(
            X=X, X_complement=X_complement, y=y, func=func
        )

        return transformed_dataset

    def to_discrete_observed(self, specific_method, **kwargs):
        """Transforms the current dataset into a Discrete Label dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {identity}
        **kwargs : dict
            not in use

        Returns
        -------
        DiscreteObservedDataset
            a newly constructed Discrete Label dataset
        """

        # Call specific transformation
        if specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("discrete")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = D_dataset.DiscreteObservedDataset(
            X=X, X_complement=X_complement, y=y, func=func
        )

        return transformed_dataset
