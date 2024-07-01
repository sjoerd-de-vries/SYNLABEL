import numpy as np
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KernelDensity

sampling_methods = [
    "uniform_independent",
    "marginal_histogram",
    "marginal_KDE",
    "multivariate_kde_scipy",
    "multivariate_kde_sklearn",
    "multivariate_imputation_without_y",
    "multivariate_imputation_with_y",
]  # independent_conditional


class Sampler:
    """This class can be used to sample new values for X_complement when using the
    Feature Hiding method to generate a Partial Ground Truth dataset.

    Parameters
    ----------
    sampling_method: string
        The method by which to sample. Allowed values: {'uniform_independent',
        'marginal_histogram', 'marginal_KDE', 'multivariate_kde_scipy',
        'multivariate_kde_sklearn', 'multivariate_imputation_without_y',
        'multivariate_imputation_with_y', 'custom' }
    X : array-like of shape (n_samples, n_features).
        The input variables available to the classification task.
    X_complement : array-like of shape (n_samples, n_features).
        The input variables not available to the classification task.
    y : array-like of shape (n_samples, 1).
        The target variable. hard labels.
    var_types: string
        The types of the different variables. Has to be a string, containing:
        ['c': Continuous 'u': Unordered (Discrete) 'o': Ordered (Discrete)].
        Has to be of the same length as the total number of features.
        Example: 'ccccucoo'

    Public Methods
    ----------
    sample(samples_per_instance, custom_sampler)
    """

    def __init__(self, sampling_method, X, X_complement, y, var_types):
        self.sampling_method = sampling_method
        self.X = X
        self.X_complement = X_complement
        self.y = y
        self.var_types = var_types

    # These sampling functions genereate values the size of
    # samples_per_instance * X_complement.shape[0]
    def sample(self, samples_per_instance, custom_sampler):
        if self.sampling_method == "uniform_independent":
            return uniform_independent_sampling(
                self.X_complement, samples_per_instance, self.var_types
            )
        elif self.sampling_method == "marginal_histogram":
            return marginal_histogram_sampling(
                self.X_complement, samples_per_instance, self.var_types
            )
        elif self.sampling_method == "marginal_KDE":
            return marginal_KDE_sampling(
                self.X_complement, samples_per_instance, self.var_types
            )
        elif self.sampling_method == "independent_conditional":
            # Not yet implemented
            pass
        elif self.sampling_method == "multivariate_kde_scipy":
            return joint_distribution_sampling(
                self.X_complement, samples_per_instance, self.var_types, method="scipy"
            )
        elif self.sampling_method == "multivariate_kde_sklearn":
            return joint_distribution_sampling(
                self.X_complement,
                samples_per_instance,
                self.var_types,
                method="sklearn",
            )
        elif self.sampling_method == "multivariate_imputation_without_y":
            return multivariate_imputation(
                self.X_complement, self.X, samples_per_instance
            )
        elif self.sampling_method == "multivariate_imputation_with_y":
            independent_data = np.hstack((self.X, self.y.reshape(-1, 1)))
            return multivariate_imputation(
                self.X_complement, independent_data, samples_per_instance
            )
        elif self.sampling_method == "custom":
            if custom_sampler is not None:
                return custom_sampler(self)
            else:
                raise Exception(
                    "Custom sampler need to be defined when sampling_method = custom"
                )
        else:
            raise (Exception("Selected sample method does not exist"))


def check_var_types(data, var_types):
    allowed_var_types = ["c", "u", "o"]

    if (len(var_types) != 0) & (len(var_types) != data.shape[1]):
        raise Exception(
            "var_types has incorrect shape."
            f"required: {data.shape[1]}. specified: {len(var_types)}."
        )

    for var_type in var_types:
        if var_type not in allowed_var_types:
            raise Exception(
                "Var type note allowed. Allowed types:"
                "[c: Continuous u : Unordered (Discrete) o : Ordered (Discrete)]"
            )


# 1 A - Uniform Independent
def uniform_independent_sampling(X_complement, samples_per_instance, var_types):
    n_columns = X_complement.shape[1]
    n_samples = X_complement.shape[0] * samples_per_instance
    sampled_data = np.zeros((n_samples, n_columns))

    # Iterate over the columns of the array
    for i in range(n_columns):
        # Ajust based on var_type
        if len(var_types) > 0:
            check_var_types(X_complement, var_types)
            if var_types[i] in ["u", "o"]:
                unique_values = np.unique(X_complement[:, i])
                sampled_data[:, i] = np.random.choice(unique_values, size=n_samples)
                # Ensure no double sampling occurs
                continue

        # Get the minimum and maximum values of the column
        minimum = X_complement[:, i].min()
        maximum = X_complement[:, i].max()

        # Sample values between the minimum and maximum
        sampled_data[:, i] = np.random.uniform(minimum, maximum, size=n_samples)

    return sampled_data


# 1 B - Independent marginal distribution
# Histogram
def marginal_histogram_sampling(X_complement, samples_per_instance, var_types):
    n_columns = X_complement.shape[1]
    n_samples = X_complement.shape[0] * samples_per_instance
    sampled_data = np.zeros((n_samples, n_columns))

    # Iterate over the columns of the array
    for i in range(n_columns):
        if len(var_types) > 0:
            check_var_types(X_complement, var_types)
            if var_types[i] in ["u", "o"]:
                unique_values, counts = np.unique(
                    X_complement[:, i], return_counts=True
                )
                frequencies = counts / X_complement.shape[0]

                sampled_data[:, i] = np.random.choice(
                    unique_values, size=n_samples, p=frequencies
                )
                # Ensure no double sampling occurs
                continue

        # Get the values of the column
        values = X_complement[:, i]

        # Calculate the marginal distribution of the column
        counts, bins = np.histogram(values, bins="auto")
        prob = counts / counts.sum()

        # Sample a value from the marginal distribution
        values = np.random.choice(bins[:-1], size=n_samples, p=prob)

        sampled_data[:, i] = values

    return sampled_data


# KDE
def marginal_KDE_sampling(
    X_complement, samples_per_instance, var_types, method="scipy"
):
    n_columns = X_complement.shape[1]
    n_samples = X_complement.shape[0] * samples_per_instance
    sampled_data = np.zeros((n_samples, n_columns))

    for i in range(n_columns):
        if len(var_types) > 0:
            check_var_types(X_complement, var_types)
            if var_types[i] in ["u", "o"]:
                unique_values, counts = np.unique(
                    X_complement[:, i], return_counts=True
                )
                frequencies = counts / X_complement.shape[0]

                sampled_data[:, i] = np.random.choice(
                    unique_values, size=n_samples, p=frequencies
                )
                # Ensure no double sampling occurs
                continue

        values = X_complement[:, i].reshape(-1, 1)

        if method == "scipy":
            kde = gaussian_kde(values.T)
            samples = kde.resample(n_samples)
            samples = samples.T
        elif method == "sklearn":
            kde = KernelDensity(kernel="gaussian")
            kde.fit(values)
            samples = kde.sample(n_samples=n_samples)

        sampled_data[:, i] = samples[:, 0]

    return sampled_data


# 1 C - Independent conditional dist
# To be implemented
# Possibly using
# https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.KDEMultivariateConditional.html#statsmodels.nonparametric.kernel_density.KDEMultivariateConditional
# Alternative - https://github.com/freelunchtheorem/Conditional_Density_Estimation
# Or single clf
# Grid sample X'
# Train CKDE on X', X
# input grid into CKDE.pdf(X') to het p(X'|X)
# multiply p(X' | X) with p(y | X' and X) aka fG to get probs. maybe normalize


# 2 A - Non-Conditional
def joint_distribution_sampling(
    X_complement, samples_per_instance, var_types, method="scipy"
):
    if len(var_types) > 0:
        check_var_types(X_complement, var_types)
        if ("u" in var_types) or ("o" in var_types):
            print("Warning: Multivariate KDE not meant for categorical variables.")

    n_samples = X_complement.shape[0] * samples_per_instance

    if method == "scipy":
        kde = gaussian_kde(X_complement.T)
        samples = kde.resample(n_samples)
        samples = samples.T
    elif method == "sklearn":
        kde = KernelDensity(kernel="gaussian")
        kde.fit(X_complement)
        samples = kde.sample(n_samples=n_samples)

    return samples


# 2 B - conditional

# https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_density.KDEMultivariateConditional.html#statsmodels.nonparametric.kernel_density.KDEMultivariateConditional
# Alternative - https://github.com/freelunchtheorem/Conditional_Density_Estimation
# Can maybe sample with:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

# Grid sample X'
# Train CKDE on X', X
# input grid into CKDE.pdf(X') to het p(X'|X)
# multiply p(X' | X) with p(y | X' and X) aka fG to get probs. maybe normalize

# Gibbs sampler or similar might be better:
# https://towardsdatascience.com/gibbs-sampling-8e4844560ae5
# To use Gibbs we need to be able to sample from p(X'|X) and p(X'|X).
# This is actually what we need to accomplish overall...


# for indep, just feed single columns x? Not sure if we need P(X, Y) read.
def gibbs_sampler(x, y, p_x_given_y, p_y_given_x, num_samples):
    # Initialize the Markov chain with random values for x and y
    x_samples = np.zeros(num_samples)
    y_samples = np.zeros(num_samples)
    x_samples[0] = np.random.choice(x)
    y_samples[0] = np.random.choice(y)

    # Run the Gibbs sampler
    for i in range(1, num_samples):
        # Sample x from its conditional distribution given the current value of y
        x_samples[i] = np.random.choice(x, p=p_x_given_y(y_samples[i - 1]))
        # Sample y from its conditional distribution given the current value of x
        y_samples[i] = np.random.choice(y, p=p_y_given_x(x_samples[i]))

    return x_samples, y_samples


# 2C MICE
def multivariate_imputation(X_complement, independent_data, samples_per_instance):
    # To do: allow for flexible estimator.
    train_data = np.hstack((X_complement, independent_data))
    missing_frame = np.full(X_complement.shape, np.nan)
    data_to_impute = np.hstack((missing_frame, independent_data))

    sampled_data = None

    for i in range(samples_per_instance):
        # Initialize the imputation model
        reg = linear_model.BayesianRidge()
        imp = IterativeImputer(
            estimator=reg, max_iter=10, random_state=i, sample_posterior=True
        )

        # Train the model and transform
        imp.fit(train_data)
        imputed_data = imp.transform(data_to_impute)

        if i == 0:
            sampled_data = imputed_data
        else:
            sampled_data = np.vstack((sampled_data, imputed_data))

    # Return only the sampled variables, not X or y
    relevant_sampled_data = sampled_data[:, : X_complement.shape[1]]

    return relevant_sampled_data
