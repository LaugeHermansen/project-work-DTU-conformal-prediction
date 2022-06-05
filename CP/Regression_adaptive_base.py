from .CP_base import Base
import numpy as np
from tqdm import tqdm

class RegressionAdaptiveBase(Base):

    def __init__(self, model, calibration_set_x, calibration_set_y, alpha, kernel, verbose = False):
        """
        instantiate class

        args: 
        -----
            kernel: the kernel function. Must be made such that
            it can take in the calibration_set_x (N_cal x M) 
            data matrix and the test_set_x (N_test x M) matrix 
            and then return the weights in an (M_test x M_cal) 
            matrix
            
            verbose: whether to print progress bar or not
        """
        self.kernel = kernel
        self.verbose = verbose
        super().__init__(model, calibration_set_x, calibration_set_y, alpha)
    
    def _quantile(self, scores):
        """
        compute the weighted 1-alpha quantile of the scores

        Args:
        ----
            scores: the scores as calculated by the score_distribution
            alpha: you know what it is.

        Returns:
        --------
            q: a function of the test points, X
        """
        n = len(scores)
        return lambda X: self._weighted_quantile(scores, X)
    
    def _weighted_quantile(self, calibration_scores, X):
        """
        compute the weighted quantile of the scores
        
        Args:
        ------
            scores: the scores of the calibration points
            X: test data
        
        Returns:
        ------
            quantiles: quantiles[i] is the weighted 1-alpha quantile of scores with
                       the i'th data point being center of the kernel.
        """

        def binary_search(cdf, i=0, j=None):
            """
            return the index of the first score where
            cdf >= self.alpha using binary search
            """
            j = len(cdf) if j == None else j
            m = int((i+j)/2)
            if i == j:  return i
            if cdf[m] < 1-self.alpha:   return binary_search(cdf, m+1, j)
            elif cdf[m] > 1-self.alpha: return binary_search(cdf, i, m)
            else: return m

        #sort scores, and init quantiles list
        ix = np.argsort(calibration_scores)
        sorted_scores = calibration_scores[ix]
        n_test = len(sorted_scores)
        quantiles = []
    
        if self.verbose:  iterable = tqdm(self.kernel(self.calibration_set_x, X), total = n_test)
        else:             iterable = iter(self.kernel(self.calibration_set_x, X))
        
        #for each data point, Xi, compute the weighted 1-alpha quantile
        for weights in iterable:
            weights = weights[ix]
            weights_cum_sum = np.cumsum(weights)
            cdf = weights_cum_sum/weights_cum_sum[-1]
            quantiles.append(sorted_scores[binary_search(cdf)])

        return np.array(quantiles)