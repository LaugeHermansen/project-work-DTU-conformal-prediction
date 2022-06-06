from .CP_base import Base
import numpy as np
from tqdm import tqdm

class RegressionAdaptiveBase(Base):

    def __init__(self, model, calibration_set_x, calibration_set_y, alpha, kernel, verbose = False):
        self.kernel = kernel
        self.verbose = verbose
        super().__init__(model, calibration_set_x, calibration_set_y, alpha)


    # def calibrate(self, calibration_set_x, calibration_set_y):
    #     """
    #     Computes the calibration quantile 
    #     Args:
    #         calibration_set_x: Calibration set inputs 
    #         calibration_set_y: The true labels/values of the calibration set

    #     Returns:
    #         Nothing. Sets self.q as the estimated 1-alpha quantile of the true distribution of scores
    #         only here, q is a function of X, the validation points.
    #     """
    #     self.calibration_set_x = calibration_set_x
    #     self.calibration_set_y = calibration_set_y

    #     scores = self.score_distribution(calibration_set_x, calibration_set_y)
    #     self.q = self._quantile(scores)

    
    def _quantile(self, scores):
        """
        compute the weighted 1-alpha quantile of the scores

        Args:
            scores: the scores as calculated by the score_distribution
            alpha: you know what it is.

        returns a function of X - the validation points as an NxM matrix
        """
        n = len(scores)
        return lambda X: self._weighted_percentile(scores, X)
    
    def _weighted_percentile(self, scores, X):

        #binary search for the index where cdf >= self.alpha
        def binary_search(cdf, i=0, j=None):
            j = len(cdf) if j == None else j
            m = int((i+j)/2)
            if i == j:  return i
            if cdf[m] < 1-self.alpha:   return binary_search(cdf, m+1, j)
            elif cdf[m] > 1-self.alpha: return binary_search(cdf, i, m)
            else: return m

        #sort scores, and init quantiles list
        ix = np.argsort(scores)
        scores = scores[ix]
        n = len(scores)
        quantiles = []
    
        if self.verbose:  iterable = tqdm(self.kernel(self.calibration_set_x, X), total = len(X))
        else:             iterable = iter(self.kernel(self.calibration_set_x, X))
        
        #for each data point, compute the weighted 1-alpha quantile
        for weights in iterable:
            weights = weights[ix]
            weights_cum_sum = np.cumsum(weights)
            cdf = weights_cum_sum/weights_cum_sum[-1]
            quantiles.append(scores[binary_search(cdf)])

            #quantiles[i] = weighted 1-alpha quantile in Xi, Xi is the i'th datapoint in validation set

        return np.array(quantiles)

