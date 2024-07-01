# Framework for generating datasets with label noise

import numpy as np
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array

from ..utils.helper_functions import (
    decision_function_generator,
    numpy_probability_check,
    one_hot_encoding,
)


class Dataset:
    """The abstract class on which all other dataset types are based.
    Includes checks for setting the values for the different components,
    as well as the _identity_transform function.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features).
        The input variables available to the classification task.
    X_complement : array-like of shape (n_samples, n_features).
        The input variables not available to the classification task.
    y : array-like of shape (n_samples, {1, n_classes}).
        The target variable. Either hard or soft labels.
    func : object
        An object (function) implementing the "predict" or "predict_proba" method.
    classes : array-like of shape (n_classes, )
        The classes of the target variable. If set for discrete datasets, it has to
        match the unique values of the target variable.
    """

    def __init__(self, **kwargs):
        self.X = kwargs.get("X", None)
        self.X_complement = kwargs.get("X_complement", None)
        self.y = kwargs.get("y", None)
        self.func = kwargs.get("func", None)
        # Can only be really set for distributed data. In the case of discrete data,
        # it has to match the inferred values from the target variable.
        self.classes = kwargs.get("classes", None)

    @property
    def X(self):
        return self._X.copy()

    @X.setter
    def X(self, value):
        self._X = check_array(value)

    @property
    def X_complement(self):
        if self._X_complement is None:
            return None
        else:
            return self._X_complement.copy()

    @X_complement.setter
    def X_complement(self, value):
        if value is not None:
            self._X_complement = check_array(value)
        else:
            self._X_complement = None

    @property
    def y(self):
        return self._y.copy()

    @y.setter
    def y(self, value):
        value_array = np.array(value)
        self._check_y(value_array)
        self._y = value_array

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, func):
        if func is not None:
            if (hasattr(func, "predict")) or (hasattr(func, "predict_proba")):
                self._func = func
            else:
                raise Exception("Function does not have predict(_proba) method")
        else:
            self._func = func

    @property
    def classes(self):
        return self._classes.copy()

    @classes.setter
    def classes(self, value):
        self._classes = self._check_classes(value)
        self._validate_class_compatability()

    def _check_y(self, value):
        if value.shape[0] != self.X.shape[0]:
            raise ValueError("X and y are different shapes")

    def get_X_y(self):
        if self.X_complement is not None:
            X = np.hstack((self.X_complement, self.X))
        else:
            X = self.X
        return X, self.y

    def _check_classes(self, value):
        pass

    def _validate_class_compatability(self):
        if hasattr(self.func, "classes_"):
            if any(self.func.classes_ != self._classes):
                raise ValueError(
                    "The classes in the dataset do not match the classes of the function."
                )

    def _identity_transform(self, target):
        """Transforms different dataset types into eachother, without changing their
        underlying data. Note that while this transformation is always possible
        "down the chain", when transforming "back up" each data set may have tighter
        contraints then the previous dataset. These are checked in the dataset
        constructor.

        Parameters
        ----------
        target : {'distributed', 'discrete'}
            The output type of the transformation.

        Returns
        -------
        self.X, self.X_complement, self.y, self.func
            All of the data necessary to initialize a dataset.
        """
        if target == "distributed":
            if self.y.ndim < 2:
                new_y = one_hot_encoding(self.y, self._classes)
            else:
                new_y = self.y
        elif target == "discrete":
            if self.y.ndim > 1:
                new_y = np.argmax(self.y, axis=1)
            else:
                new_y = self.y

        return self.X, self.X_complement, new_y, self.func


class DistributedDataset(Dataset):
    """The class on which the PG and LD dataset types are based.
    Includes checks for setting the values for the different components,
    as well as the _identity_transform and _select_most_probable functions.

    Parameters
    ----------
    See "Dataset"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_y(self, value):
        super()._check_y(value)
        if len(value.shape) <= 1:
            raise ValueError(
                "y has one ore fewer dimensions, which is not allowed for"
                "a DistributedDataset"
            )
        numpy_probability_check(value)

    def _check_classes(self, value):

        if value is None:
            classes = np.arange(self.y.shape[1])
        elif self.y.shape[1] != len(value):
            raise ValueError("Classes have to match the shape of the target variable.")
        else:
            classes = value
        return classes


class DiscreteDataset(Dataset):
    """The class on which the G and D dataset types are based.
    Includes checks for setting the values for the different components,
    as well as the _identity_transform and _transform_to_distribution functions.

    Parameters
    ----------
    See "Dataset"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_y(self, value):
        super()._check_y(value)

        if len(value.shape) != 1:
            raise ValueError(
                "y has more than one dimension, which is not allowed for"
                "a DicreteDataset"
            )

        # Ensure that target y is of a non-regression type
        check_classification_targets(value)

    def _check_classes(self, value):

        classes_in_data = np.unique(self.y)
        if value is not None:
            if len(classes_in_data) != len(value):
                print("Warning: not all known classes are present in the data")
            elif any(classes_in_data != value):
                raise ValueError(
                    "The classes in the data do not match the given classes."
                )
        else:
            value = classes_in_data
        return value


class ObservedDataset(Dataset):
    """The class on which the LD and D dataset types are based.
    Includes checks for setting the values for the different components,
    as well as the _identity_transform, _transform_to_partial_ground_truth
    and _transform_to_ground_truth functions.

    Parameters
    ----------
    See "Dataset"
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform_to_partial_ground_truth(self, function=None):
        if function is None:
            print("No function defined, self.func is used.")
            function = self.func

        if hasattr(function, "predict_proba"):
            y = function.predict_proba(self.X)
        else:
            raise Exception("Function does not have predict(_proba) method")
        return self.X, self.X_complement, y, function

    def _transform_to_ground_truth(
        self, function=None, decision_function=None, classes=None
    ):
        if function is None:
            print("No function defined, self.func is used.")
            function = self.func

        if hasattr(function, "predict"):
            y = function.predict(self.X)
        elif hasattr(function, "predict_proba"):
            if decision_function is None:
                print("No decision function defined, numpy argmax is used.")
                decision_function = lambda x: np.argmax(x)
            function = decision_function_generator(function, decision_function)
            y_indices = function.predict(self.X)
            y = np.array(classes)[y_indices]
        else:
            raise Exception("Function does not have predict(_proba) method")

        return self.X, self.X_complement, y, function
