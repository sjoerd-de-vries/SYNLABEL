from . import distributed_observed as LD_dataset
from . import ground_truth as G_dataset
from . import partial_ground_truth as PG_dataset
from .abstract_datasets import DiscreteDataset, ObservedDataset
from .expert_data import ExpertData


class DiscreteObservedDataset(DiscreteDataset, ObservedDataset, ExpertData):
    """The Discrete Label (D) class. Inherits from DiscreteDataset, ObservedDataset
    and ExpertData.

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
    learn_function(clf)
    to_ground_truth(specific_method, **kwargs)
    to_partial_ground_truth(specific_method, **kwargs)
    to_distributed_observed(specific_method, **kwargs)


    See specific documentation of the methods for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def learn_function(self, clf):
        """Learns a function f_O based on self.X and self.y.
        Sets this function to self.func.

        Parameters
        ----------
        clf : classifier
            has to implement predict or predict_proba
        """
        clf.fit(self.X, self.y)

        self.func = clf
