from .CP_base import Base
import numpy as np
from tqdm import tqdm

class RegressionBase(Base):

    def _in_pred_set(self, prediction_sets, y):
        return np.array(list(map(lambda a: a[1][0] <= a[0] <= a[1][1], zip(y, prediction_sets))))

    def _pred_set_sizes(self, prediction_sets):
        return prediction_sets[:,1] - prediction_sets[:,0]

    # def __init__(self, model, calibration_set_x, calibration_set_y, alpha, call_function_name = None, name = None, kernel = None, verbose = False):
    #     """
    #     instantiate class

    #     args: 
    #     -----
    #         kernel: the kernel function. Must be made such that
    #         it can take in the calibration_set_x (N_cal x M) 
    #         data matrix and the test_set_x (N_test x M) matrix 
    #         and then return a generator of the weights for each 
    #         sample
            
    #         verbose: whether to print progress bar or not
    #     """

    #     self.adaptive = kernel != None
    #     self.kernel = kernel
    #     self.verbose = verbose
        
    #     if   name == None and not self.adaptive: name = self.__class__.__name__.replace("Adaptive", "").replace("adaptive", "")
    #     elif name == None and self.adaptive: name = f"{self.__class__.__name__}, kernel: {self.kernel.__name__}"

    #     super().__init__(model, calibration_set_x, calibration_set_y, alpha, call_function_name, name)
    
    # def _quantile(self, calibration_scores):
    #     """
    #     compute the weighted 1-alpha quantile of the scores

    #     Args:
    #     ----
    #         scores: the scores as calculated by the score_distribution

    #     Returns:
    #     --------
    #         q: the empirical 1-alpha quantile of calibration data
    #             as a function of the test points, X
    #             the q function returns the estimated mean effective sample size as well                                                                      
    #     """
    #     if self.adaptive:
    #         return lambda X: self._weighted_quantile(calibration_scores, X)
    #     else:
    #         #the level on which we take the quantile of the scores
    #         level = np.ceil((self.n_cal+1)*(1-self.alpha))/self.n_cal
    #         q = np.quantile(calibration_scores, level)
    #         return lambda X: (np.ones(len(X))*q, self.n_cal)

    # def _weighted_quantile(self, calibration_scores, X):
    #     """
    #     compute the weighted quantile of the scores
        
    #     Args:
    #     ------
    #         scores: the scores of the calibration points
    #         X: test data
        
    #     Returns:
    #     ------
    #         quantiles: quantiles[i] is the weighted 1-alpha quantile of scores with
    #                    the i'th data point being center of the kernel.
    #     """

    #     def binary_search(cdf, i=0, j=None):
    #         """
    #         return the index of the first score where
    #         cdf >= self.alpha using binary search
    #         """
    #         j = len(cdf) if j == None else j
    #         m = int((i+j)/2)
    #         if i == j:  return i
    #         if cdf[m] < 1-self.alpha:   return binary_search(cdf, m+1, j)
    #         elif cdf[m] > 1-self.alpha: return binary_search(cdf, i, m)
    #         else: return m

    #     #sort scores, and init quantiles list
    #     ix = np.argsort(calibration_scores)
    #     sorted_scores = calibration_scores[ix]
    #     n_test = len(X)
    #     quantiles = []
    #     effective_sample_sizes = []
    
    #     if self.verbose:  print(f"Fitting adaptive quantiles - {self.name}")
    #     if self.verbose:  iterable = tqdm(self.kernel(self.calibration_set_x, X), total = n_test)
    #     else:             iterable = iter(self.kernel(self.calibration_set_x, X))
        
    #     #for each data point, Xi, compute the weighted 1-alpha quantile
    #     for i, kernel_values in enumerate(iterable):
    #         kernel_values = kernel_values[ix]
    #         kernel_values_cum_sum = np.cumsum(kernel_values)
    #         if np.sum(kernel_values) == 0:
    #             crappy_data_point = X[i]
    #             raise ValueError('Encounteret test point where kernel(x) = 0 for all x in calibration set')
    #         effective_sample_sizes.append(np.sum(kernel_values)**2/np.sum(kernel_values**2))
    #         if kernel_values_cum_sum[-1] == 0:  cdf = np.arange(1,1+self.n_cal)/self.n_cal
    #         else:                         cdf = kernel_values_cum_sum/kernel_values_cum_sum[-1]
    #         quantiles.append(sorted_scores[binary_search(cdf)])

    #     return np.array(quantiles), np.array(effective_sample_sizes)

    
    # def evaluate_coverage(self, X, y):
    #     """
    #     Evaluate epirical coverage on test data points.
        
    #     Args:
    #     ------
    #         X: the features of the test data points
    #         y: the true labels/values of the test data points
        
    #     Returns:
    #     --------
    #         The empirical coverage of the test data points
    #         The prediction
    #     """
    #     y_preds, pred_intervals, effective_sample_sizes = self.predict(X)
    #     in_pred_set = np.array(list(map(lambda a: a[1][0] <= a[0] <= a[1][1], zip(y, pred_intervals))))
    #     empirical_coverage = np.mean(in_pred_set)
    #     return y_preds.squeeze(), pred_intervals, in_pred_set, empirical_coverage, effective_sample_sizes