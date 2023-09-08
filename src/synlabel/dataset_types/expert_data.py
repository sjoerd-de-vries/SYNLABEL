import numpy as np
from sklearn.utils.validation import check_array

from .abstract_datasets import Dataset


class ExpertData(Dataset):
    """The Expert (E) class. Inherits from DataSet. An abstract dataset, parent of
    DiscreteObservedDataset (D) and DistributedObservedDataset (LD). Fascilitates
    the construction of an explicit labeling process by annotators.

    Parameters
    ----------
    X_E : array-like of shape (n_samples, n_features).
        The input variables available to the annotation task.
        A combination of X_E_from_X and X_E_from_X_complement.
    func_E : object
        An object (function) implementing the "predict" or "predict_proba" method.
        Used to generate a label by expert decision.
    X_E_from_X : array-like of shape (n_samples, n_features_X).
        The input variables available to the classification task, as seen
        by the annotator.
    X_E_from_X_complement : array-like of shape (n_samples, n_features_X_complement).
        The input variables not available to the classification task, as seen by the
        annotator.

    Public Methods
    ----------
    transform_X_complement_to_expert(transformation, update_X_E)
    transform_X_to_expert(transformation, update_X_E)

    See specific documentation of the methods for more information.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.X_E = kwargs.get("X_E", None)
        self.func_E = kwargs.get("func_E", None)
        self.X_E_from_X = kwargs.get("X_E_from_X", None)
        self.X_E_from_X_complement = kwargs.get("X_E_from_X_complement", None)

    @property
    def X_E(self):
        return self._X_E

    @X_E.setter
    def X_E(self, value):
        if value is not None:
            self._X_E = check_array(value)
        else:
            self._X_E = None

    @property
    def X_E_from_X(self):
        return self._X_E_from_X

    @X_E_from_X.setter
    def X_E_from_X(self, value):
        if value is not None:
            self._X_E_from_X = check_array(value)
        else:
            self._X_E_from_X = None

    @property
    def X_E_from_X_complement(self):
        return self._X_E_from_X_complement

    @X_E_from_X_complement.setter
    def X_E_from_X_complement(self, value):
        if value is not None:
            self._X_E_from_X_complement = check_array(value)
        else:
            self._X_E_from_X_complement = None

    @property
    def func_E(self):
        return self._func_E

    @func_E.setter
    def func_E(self, func):
        if func is not None:
            if (hasattr(func, "predict")) or (hasattr(func, "predict_proba")):
                self._func_E = func
            else:
                raise Exception("Function does not have predict(_proba) method")
        else:
            self._func_E = func

    def transform_X_complement_to_expert(self, transformation, update_X_E):
        """Constructs X_E_from_X_complement based on self.X_complement, by applying
        a transformation supplied by the user.

        Parameters
        ----------
        transformation : function
            function to be applied to the data.
        update_X_E : bool
            if True, self.X_E is constructed from self.X_E_from_X_complement
            and self.X_E_from_X.

        """
        try:
            self.X_E_from_X_complement = transformation(self.X)
        except Exception:
            raise Exception("transformation could not be applied")

        # Set X_E to be combination of X_E_available and X_E_complement
        if update_X_E:
            self.X_E = np.hstack((self.X_E_from_X_complement, self.X_E_from_X))

    def transform_X_to_expert(self, transformation, update_X_E):
        """Constructs X_E_from_X based on self.X, by applying
        a transformation supplied by the user.

        Parameters
        ----------
        transformation : function
            function to be applied to the data.
        update_X_E : bool
            if True, self.X_E is constructed from self.X_E_from_X_complement
            and self.X_E_from_X.

        """
        try:
            self.X_E_from_X = transformation(self.X)
        except Exception:
            raise Exception("transformation could not be applied")

        # Set X_E to be combination of X_E_available and X_E_complement
        if update_X_E:
            self.X_E = np.hstack((self.X_E_from_X_complement, self.X_E_from_X))
