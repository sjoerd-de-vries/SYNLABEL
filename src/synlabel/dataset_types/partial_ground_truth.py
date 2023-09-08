import numpy as np

from ..utils.helper_functions import decision_function_generator
from . import discrete_observed as D_dataset
from . import distributed_observed as LD_dataset
from . import ground_truth as G_dataset
from .abstract_datasets import DistributedDataset


class PartialGroundTruthDataset(DistributedDataset):
    """The Partial Ground Truth (PG) class. Inherits from DistributedDataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features).
        The input variables available to the classification task.
    X_complement : array-like of shape (n_samples, n_features).
        The input variables not available to the classification task.
    y : array-like of shape (n_samples, n_classes).
        The target variable. Soft labels.
    func : object
        An object (function) implementing the "predict" or "predict_proba" method.

    Public Methods
    ----------
    to_ground_truth(specific_method, **kwargs)
    to_distributed_observed(specific_method, **kwargs)
    to_discrete_observed(specific_method, **kwargs)

    See specific documentation of the methods for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _deterministic_decision(self, decision_function):
        if decision_function is None:
            print("No decision function defined, numpy argmax is used.")
            decision_function = lambda x: np.argmax(x)

        func = decision_function_generator(self.func, decision_function)
        try:
            y = func.predict(self.X)
        except Exception:
            raise Exception("transformation could not be applied")

        X, X_complement = self.X, self.X_complement
        return X, X_complement, y, func

    def _transform_x(self, transformation):
        try:
            X = transformation(self.X)
        except Exception:
            raise Exception("transformation could not be applied")

        X_complement, y, func = self.X_complement, self.y, self.func
        return X, X_complement, y, func

    def _transform_y(self, transformation):
        try:
            y = transformation(self.y)
        except Exception:
            raise Exception("transformation could not be applied")

        X, X_complement, func = self.X, self.X_complement, self.func
        return X, X_complement, y, func

    def _transform_x_and_y(self, transformation):
        try:
            X, y = transformation(self.X, self.y)
        except Exception:
            raise Exception("transformation could not be applied")

        X_complement, func = self.X_complement, self.func
        return X, X_complement, y, func

    def _transform_x_complement(self, transformation):
        try:
            X_complement = transformation(self.X_complement)
        except Exception:
            raise Exception("transformation could not be applied")

        X, y, func = self.X, self.y, self.func
        return X, X_complement, y, func

    def to_ground_truth(self, specific_method, **kwargs):
        """Transforms the current dataset into a Ground Truth dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {determinstic_decision, identity}
        **kwargs : dict
            decision_function (specific_method=function_based):
                the function used to discretize the results.
                if not specified, np.argmax is used.

        Returns
        -------
        GroundTruthDataset
            a newly constructed Ground Truth dataset
        """

        # Call specific transformation
        if specific_method == "determinstic_decision":
            decision_function = kwargs.get("decision_function", None)
            X, X_complement, y, func = self._deterministic_decision(decision_function)
        elif specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("discrete")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = G_dataset.GroundTruthDataset(X=X, y=y, func=func)

        return transformed_dataset

    def to_distributed_observed(self, specific_method, **kwargs):
        """Transforms the current dataset into a Label Distribution dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {transform_x, transform_y, transform_x_and_y,
            transform_x_complement, identity}
        **kwargs : dict
            transformation (specific_method=all, except 'identity'):
                transformation to be applied to the specified data

        Returns
        -------
        DistributedObservedDataset
            a newly constructed Label Distribution dataset
        """
        # Call specific transformation
        if specific_method == "transform_x":
            transformation = kwargs["transformation"]
            X, X_complement, y, func = self._transform_x(transformation)
        # Call specific transformation
        elif specific_method == "transform_y":
            transformation = kwargs["transformation"]
            X, X_complement, y, func = self._transform_y(transformation)
        # Call specific transformation
        elif specific_method == "transform_x_and_y":
            transformation = kwargs["transformation"]
            X, X_complement, y, func = self._transform_x_and_y(transformation)
        # Call specific transformation
        elif specific_method == "transform_x_complement":
            transformation = kwargs["transformation"]
            X, X_complement, y, func = self._transform_x_complement(transformation)
        elif specific_method == "identity":
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
