import numpy as np

from . import discrete_observed as D_dataset
from . import ground_truth as G_dataset
from . import partial_ground_truth as PG_dataset
from .abstract_datasets import DistributedDataset, ObservedDataset
from .expert_data import ExpertData

rng = np.random.default_rng()


class DistributedObservedDataset(DistributedDataset, ObservedDataset, ExpertData):
    """The Label Distribution (LD) class. Inherits from DistributedDataset,
    ObservedDataset and ExpertData.

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
    to_partial_ground_truth(specific_method, **kwargs)
    to_discrete_observed(specific_method, **kwargs)

    See specific documentation of the methods for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sample(self):
        y_shape = self.y.shape[0]
        y = np.zeros(y_shape)
        for i in range(y_shape):
            y[i] = rng.choice(self.y.shape[1], p=self.y[i, :])

        X, X_complement, func = self.X, self.X_complement, self.func
        return X, X_complement, y, func

    def _argmax(self):
        y = np.argmax(self.y, axis=1)

        X, X_complement, func = self.X, self.X_complement, self.func
        return X, X_complement, y, func

    def _decision_function(self, transformation):
        try:
            y = transformation(self.y)
        except Exception:
            raise Exception("transformation could not be applied")

        X, X_complement, func = self.X, self.X_complement, self.func
        return X, X_complement, y, func

    def to_ground_truth(self, specific_method, **kwargs):
        """Transforms the current dataset into a Ground Truth dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {function_based, identity}
        **kwargs : dict
            function (specific_method=function_based):
                the function to be set as f_G.
                if not specified, self.func is used.
            decision_function (specific_method=function_based):
                the function used to discretize the results.
                if not specified, np.argmax is used.

        Returns
        -------
        GroundTruthDataset
            a newly constructed Ground Truth dataset
        """
        # Call specific transformation
        if specific_method == "function_based":
            transformation = kwargs.get("function", None)
            decision_function = kwargs.get("decision_function", None)
            X, X_complement, y, func = super()._transform_to_ground_truth(
                transformation, decision_function
            )
        elif specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("discrete")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = G_dataset.GroundTruthDataset(X=X, y=y, func=func)

        return transformed_dataset

    def to_partial_ground_truth(self, specific_method, **kwargs):
        """Transforms the current dataset into a Partial Ground Truth dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {function_based, identity}
        **kwargs : dict
            function (specific_method=function_based):
                the function to be set as f_G.
                if not specified, self.func is used.

        Returns
        -------
        PartialGroundTruthDataset
            a newly constructed Partial Ground Truth dataset
        """

        # Call specific transformation
        if specific_method == "function_based":
            transformation = kwargs.get("function", None)
            X, X_complement, y, func = super()._transform_to_partial_ground_truth(
                transformation
            )
        elif specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("distributed")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = PG_dataset.PartialGroundTruthDataset(
            X=X, X_complement=X_complement, y=y, func=func
        )

        return transformed_dataset

    def to_discrete_observed(self, specific_method, **kwargs):
        """Transforms the current dataset into a Discrete Label dataset.

        Parameters
        ----------
        specific_method : string
            the method by which the transformation takes place.
            options: {decision_function, sample, plurality_decision, identity}
        **kwargs : dict
            decision_function (specific_method=decision_function):
                the function used to discretize y.

        Returns
        -------
        DiscreteObservedDataset
            a newly constructed Discrete Label dataset
        """

        # Call specific transformation
        if specific_method == "decision_function":
            decision_function = kwargs["decision_function"]
            X, X_complement, y, func = self._decision_function(decision_function)
        elif specific_method == "sample":
            X, X_complement, y, func = self._sample()
        elif specific_method == "plurality_decision":
            X, X_complement, y, func = self._argmax()
        elif specific_method == "identity":
            X, X_complement, y, func = super()._identity_transform("discrete")
        else:
            raise Exception(f"Specific method: {specific_method} not available")

        transformed_dataset = D_dataset.DiscreteObservedDataset(
            X=X, X_complement=X_complement, y=y, func=func
        )

        return transformed_dataset
