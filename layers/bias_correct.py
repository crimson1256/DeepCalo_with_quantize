import numpy as np
import itertools
from scipy.interpolate import interp1d, bisplrep, interp2d
from scipy.interpolate.dfitpack import bispeu


class BiasCorrect():
    """Bias correction class"""

    def __init__(self, error_func=None):
        self.fitted = False
        self.intp = None

    def error_func(self, y_true, y_pred):
        return y_pred - y_true

    def fit(self, y_pred, y_true, param=None, n_bins=50):

        # Simple bias correction
        self.simple_bias = np.median(y_pred - y_true)

        # Use cubic splines to correct as a function of a parameter param
        if param is not None:

            # Get x,y for interp1d
            # x: Centers of binned param values
            hist, edges = np.histogram(param, bins=n_bins, density=False)
            centers = edges[:-1] + 0.5*np.diff(edges)

            # y: median error in that bin (i.e., the bias)
            error = self.error_func(y_true, y_pred)
            medians = np.zeros(n_bins)
            for i in range(n_bins):
                mask = (param >= edges[i]) & (param < edges[i+1])
                medians[i] = np.median(error[mask])

            self.intp = interp1d(centers, medians, kind='cubic', fill_value='extrapolate') # 0.0, bounds_error=False)

        self.fitted = True

    def transform(self, y_pred, param=None):

        if not self.fitted:
            print('Please call the fit method before calling the transform method.')
            return None
        else:
            if param is None:
                return y_pred - self.simple_bias
            else:
                return y_pred - self.intp(param)


class BiasCorrect2D():
    """Bias correction class"""

    def __init__(self):
        self.fitted = False

    def fit(self, y_true, y_pred, params, bins, smoothing_factor=None,
            use_weights=False):
        '''
        y_true : Numpy array containing true values with shape (N,).
        y_pred : Numpy array containing predicted values with shape (N,).
        params : List containing numpy arrays with the values that the spline
                 should be fitted as a function of, each with shape (N,).
        bins : List containing a Numpy array with bin edges for each param.
        smoothing_factor : A non-negative smoothing factor. None (default) will
                           use sqrt(m)-2*m (see bisplrep docs)
        use_weights : Boolean or str indicating whether to use weights in fit.
                      Strings can be 'sqrt' or 'log'.
        '''

        assert(type(params)==list)
        n_points = params[0].shape[0]
        n_params = len(params)
        n_bins = [bin.shape[0]-1 for bin in bins]
        n_spline_points = np.prod(n_bins)
        print(n_spline_points)

        # -------------------
        # Create X for spline
        # -------------------
        # Create multi-dimensional histogram over params
        hist, edges = np.histogramdd(np.array(params).T, bins=bins)
        centers = np.array([np.array(edges[i])[:-1] + 0.5*np.diff(np.array(edges[i])[:]) for i in range(n_params)])

        # Convert hist_ij to (p0[i],p1[j]) (if n_params==2) for all ij
        # This will be our X that the spline should fit to
        X = np.array(list(itertools.product(*centers))) # Shape: (n_spline_points, n_params)

        # Create weights for spline fit (inverse of std of errors in y)
        if use_weights:
            if use_weights=='log':
                weights = np.log(hist + 2).reshape(n_spline_points)
            else:
                weights = np.sqrt(hist + 1).reshape(n_spline_points)
        else:
            weights = None

        # -------------------
        # Create y for spline
        # -------------------
        # y is the median error in each bin
        # y should have the same first dimension as X
        error = y_pred - y_true
        y = np.zeros(n_spline_points) # Shape: (n_spline_points,)

        for i,bin_indices in enumerate(itertools.product(*[range(n_bin) for n_bin in n_bins])):
            mask = np.ones(n_points).astype(np.bool)
            for j in range(n_params):
                mask *= (params[j] >= edges[j][bin_indices[j]]) & (params[j] < edges[j][bin_indices[j]+1])

            # Write to y[i]
            if np.all(mask==False):
                median_error_in_bin = 0
            else:
                median_error_in_bin = np.median(error[mask])
            y[i] = median_error_in_bin

        # Fit spline
        # TODO: Include fill_value=0
        self.intp = bisplrep(X[:,0], X[:,1], y, w=weights, s=smoothing_factor)

        self.fitted = True

    def transform(self, y_pred, params):

        if not self.fitted:
            print('Please call the fit method before calling the transform method.')
            return None
        else:
            correction, _ = bispeu(*self.intp, *params)
            return y_pred - correction
