# Numerical packages
import numpy as np
from KDEpy.FFTKDE import FFTKDE
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


class KDEBoosting(object):
    """Class for performing boosting of the Kernel Density Estimate of a multi-
    dimensional data array.
    """

    def __init__(self, data, k_iterations=1, standardize_data=True, n_jobs=None,
                 verbose=False, **kde_kwargs):
        """The class is initialized by providing the data and the number of
        boosting iterations to perform.

        :param data: [numpy array] The data on which to run the algorithm. It is
            of shape [_n_samples x _n_dimensions]. The data is stored in the object
            so that boosting iterations can be added without the need providing
            the data again.
        :param k_iterations: [int > 0] The number of boosting iterations to run
            at initialization. The object can be initialized with k_iterations =
            0 and the algorithm run after. If initialized with k_iterations = 0,
            the weights are trivially 1 / n, where n is the number of samples in
            the data.
        :param standardize_data: [bool] Whether to standardize the data to 0 mean
            and 1 standard deviation. It equals True by default.
        :param n_jobs: [int or None] The number of cores to use for computing the
            leave-one-out cross-validated KDE. It is convenient to set this value
            to -1 (to use all cores). However, it is None by default, which
            generally equals to serial processing.
        :param verbose: [bool] Whether to print joblib messages during leave-one-
            out estimation.
        :param kde_kwargs: [dict] Pass attributes to KDE algorithm.
            :param bw: [float or str] The bandwidth parameter that will be passed
                to KDEpy.FFTKDE. It can be 'Scott', 'Silverman', 'ISJ' or a float.
            :param n_grid_points: [int] Number of points which will be used to
                create the equidistant evaluation grid.

        The following outputs are stored as attributes of the object:
        :return weights: [numpy array] The weights for the highest boosting
            iteration. It has shape [n_samples x 1].
        :return normalized_weights: [numpy array] The normalized weights for the
            highest boosting iteration. It has shape [n_samples x 1].
        :return intermediate_weights: [numpy array] The weights for all the
            iterations below the highest. It has shape [n_samples x k_iterations - 1].
        """

        # Store data and whether to normalize them
        self._standardize_data = standardize_data
        self.data = data
        # Extract other parameters for KDE
        self._kernel_bandwidth = kde_kwargs.pop('bw', 'silverman')
        self._n_grid_points = kde_kwargs.pop('n_grid_points', None)

        # Store parameters for the KDE
        self._kde_kwargs = kde_kwargs

        # Get data dimensionality
        self._n_samples, self._n_dimensions = self.data.shape
        self._data_type = data.dtype

        # Initialize output variables
        self.normalized_weights = None
        self.weights = np.ones((self._n_samples, 1), dtype=self._data_type) / self._n_samples
        self.intermediate_weights = np.zeros((self._n_samples, 0), dtype=self._data_type) * np.nan
        self._pdf_kde = np.zeros((self._n_samples, 0), dtype=self._data_type) * np.nan
        self._loo_kde = np.zeros((self._n_samples, 0), dtype=self._data_type) * np.nan

        # Run boosting algorithm
        self.k_iterations = 0
        self._n_jobs = n_jobs
        self._verbose = verbose
        if k_iterations > 0:
            self.boost(k_iterations)
        else:
            self.k_iterations = k_iterations


    def boost(self, up_to_iteration_k):
        """The main algorithm, as proposed by Di Marzio and Taylor.

        :param up_to_iteration_k: [int] Number of iterations of the algorithm to
            run. If the user is explicitly calling this method after
            initialization, only the additional iterations will be run. For
            example, if the user has initialized the object with 2 iterations and
            then calls this method with `up_to_iteration_k` 3, one more iteration
            will be run to reach the input value of 3. If, instead, the value is
            2 or less, the function returns immediately and no action is performed.
        """

        # Get the number of iterations already run
        previous_k_iterations = self.k_iterations
        # Get the number of iterations to run
        new_iterations = up_to_iteration_k - previous_k_iterations
        if new_iterations < 1:
            return

        # Allocate weights (initialized as uniform distribution), and arrays to
        # keep track of PDF and leave-one-out PDF at each boosting iteration
        w = np.ones((self._n_samples, new_iterations), dtype=self._data_type) / self._n_samples
        pdf_kde = np.zeros((self._n_samples, new_iterations), dtype=self._data_type) * np.nan
        loo_kde = np.zeros((self._n_samples, new_iterations), dtype=self._data_type) * np.nan
        # Concatenate these arrays to what has been already calculated
        w = np.hstack((self.intermediate_weights, w))
        self._pdf_kde = np.hstack((self._pdf_kde, pdf_kde))
        self._loo_kde = np.hstack((self._loo_kde, loo_kde))

        for i_iter in range(new_iterations):
            k = previous_k_iterations + i_iter
            # Adjust weights in all iterations after the first one
            if k > 0:
                # In the original paper, the log odds ratio was considered, but
                # here both PDFs are already log(densities). Therefore, take
                # the difference, because log(x/y) = log(x) - log(y).
                w[:, k] = w[:, k - 1] + (self._pdf_kde[:, k - 1] - self._loo_kde[:, k - 1])
                # Make sure weights sum up to 1
                w[:, k] /= np.sum(w[:, k])
            # Reshape weights
            weights = np.atleast_2d(w[:, k]).transpose()

            # Compute PDF
            self._pdf_kde[:, k] = _compute_kde(data=self.data, bw=self._kernel_bandwidth,
                                               weights=weights, standardize=self._standardize_data,
                                               n_grid_points=self._n_grid_points)
            # Compute leave-one-out PDF
            self._loo_kde[:, k] = _compute_loo_kde(data=self.data, bw=self._kernel_bandwidth,
                                                   weights=weights,
                                                   standardize=self._standardize_data,
                                                   n_grid_points=self._n_grid_points,
                                                   n_jobs=self._n_jobs,
                                                   verbose=self._verbose).ravel()

        # Do final adjustment of weights (log odds ratio, and sum up to 1)
        weights = w[:, -1] + (self._pdf_kde[:, -1] - self._loo_kde[:, -1])
        weights /= np.sum(weights)

        # Store weights
        self.weights = weights
        self.intermediate_weights = w
        # Store number of iterations performed
        self.k_iterations = max(self.k_iterations, up_to_iteration_k)
        # Store normalized weights
        self.normalize_weights(original_implementation=True)


    def normalize_weights(self, original_implementation=True):
        if original_implementation:
            # Invert and normalize the final probability weights (this creates
            # asymmetries in the final distribution)
            normalized_weights = 1. / self.weights

        else:
            # Flip weights around value of uniform distribution and cap negative
            # weights to 0
            normalized_weights = 2 * (1 / self._n_samples) - self.weights
            neg_idx = np.where(normalized_weights < 0)[0]
            normalized_weights[neg_idx] = 1. / self.weights[neg_idx] / np.sum(1. / self.weights)
            normalized_weights[normalized_weights < 0] = 0

        # Make sure weights sum up to 1
        normalized_weights /= np.sum(normalized_weights)

        self.normalized_weights = normalized_weights


def _compute_loo_kde(data, bw, weights, standardize, n_grid_points, n_jobs, verbose):
    """Utility function to run leave-one-out KDE with joblib, which will distribute
    the computation of each sample to each core.

    :param data: [numpy array] The data on which to run the algorithm. It is
        of shape [_n_samples x _n_dimensions]. The data is stored in the object
        so that boosting iterations can be added without the need providing
        the data again.
    :param bw: [float] The bandwidth parameter that will be passed to
        sklearn.neighbors.kde.KernelDensity.
    :param weights: [numpy array] The weights for the highest boosting
        iteration. It has shape [n_samples x 1].
    :param standardize: [bool] Whether to standardize the data to mean 0 and
        standard deviation 1.
    :param n_jobs: [int or None] The number of cores to use for computing the
        leave-one-out cross-validated KDE. It is convenient to set this value
        to -1 (to use all cores). However, it is None by default, which
        generally equals to serial processing.

    :return: [numpy array] Log density estimation of each left-out sample,
        given the KDE computed on all but this one sample.
    """
    return np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_compute_kde)(data=data[train_index, :], bw=bw,
                                  weights=weights[train_index, :],
                                  standardize=standardize,
                                  n_grid_points=n_grid_points,
                                  return_pdf_at=data[test_index, :])
            for train_index, test_index in LeaveOneOut().split(data)))


def _compute_kde(data, bw, weights, standardize, n_grid_points, return_pdf_at=None):
    """Compute KDE and return log densities.

    :param data: [numpy array] The data on which to run the algorithm. It is
        of shape [_n_samples x _n_dimensions]. The data is stored in the object
        so that boosting iterations can be added without the need providing
        the data again.
    :param bw: [float] The bandwidth parameter that will be passed to
        sklearn.neighbors.kde.KernelDensity.
    :param weights: [numpy array] The weights for the highest boosting
        iteration. It has shape [n_samples x 1].
    :param standardize: [bool] Whether to standardize the data to mean 0 and
        standard deviation 1.
    :param n_grid_points: [int] Number of grid points on an equidistant grid on
        which to evaluate the KDE.
    :param return_pdf_at: [numpy array] The data on which to calculate the log
        density given the KDE of `data`. If None, the log density of all `data`
        samples will be returned. In the context of leave-one-out KDE, it is
        the left-out sample.
    :return: [numpy array] Log density estimation of each left-out sample,
        given the KDE computed on all but this one sample.
    """
    # Set data points at which to return the PDF, and scale them
    if return_pdf_at is None:
        return_pdf_at = data.copy()

    # Standardize data, if requested
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
        data = scaler.transform(data)
        if standardize:
            return_pdf_at = scaler.transform(return_pdf_at)

    # Estimate KDE over a grid
    kde_estimator = FFTKDE(kernel='gaussian', bw=bw).fit(data, weights=weights.ravel())
    kde_grid, kde = kde_estimator.evaluate(grid_points=n_grid_points)

    # Get number of data dimensions
    n_dims = data.shape[1]
    if n_dims == 1:
        kde_grid = kde_grid.reshape(-1, 1)

    # Get grid shape and unique grid axes
    grid_axes = [np.atleast_2d(np.unique(kde_grid[:, i])).T for i in range(n_dims)]

    # Ge the number of bins along each dimension
    num_grid_points = [len(i) for i in grid_axes]

    # Find coordinates of data points on grid
    coords = np.array([np.abs(grid_axes[d] - np.atleast_2d(return_pdf_at[:, d])).argmin(axis=0) for d in range(n_dims)]).T
    bin_indices = np.ravel_multi_index(multi_index=coords.T, dims=num_grid_points)
    # Take the log of the values at those positions
    values = kde[bin_indices].ravel()

    return values
