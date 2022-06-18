from dataclasses import dataclass
import numpy as np
from tqdm import tqdm


class Base():

    # reguired methods to be specified for CP to work

    def score_distribution(self):
        """
        Compute the scores of the calibration set.

        Args:
        -----
            Takes nothing as the calibration set is in the init

        Returns:
        -------
            All scores of the labels of the calibration set
        """
        raise NotImplementedError()

    def score(self, X):
        """
        Compute score of new data

        Args:
        -----
            X: test inputs NxM matrix

        Returns:
        -------
            Scores of X and all labels in label space.
        """
        raise NotImplementedError()


    def predict(self, X):
        """
        Compute prediction set for new data points

        Note that self.q is a function that takes in test points, X,
        and returns a tuple containing:
            self.q(X)[0]: the empirical 1-alpha quantiles of calibration scores
                          evaluated at each test point in X,
                          as a 1D np.array of length n_test
            self.q(X)[1]: effective sample sizes, X, N_test x 1 np.array

        Args:
        -----
            X: The new data points
            
        Returns:
        --------
            y_pred: underlying model prediction or output
            pred_sets: Prediction sets - one for each test point.
            effective_sample_sizes: N_text x 1 np.array (returned from self.q)
        """
        raise NotImplementedError()


    def _in_pred_set(self, prediction_sets, y):
        """
        Compute the data points where the true label is in the prediction set

        Args:
        -----
            prediction_sets: 
            y:
        
        Returns:
        --------
            Return 1D a boolean array of length Ntest
        """
        raise NotImplementedError()
    
    def _pred_set_sizes(self, prediction_sets):
        """
        Compute the sizes of the prediction sets

        Args:
        -----
            prediction_sets:
        
        Returns:
        --------
            Return 1D float array of length Ntest
        """
        raise NotImplementedError()

    

    # Keep your mits off my grub - means that the methods below should be general
    # and there is no need to change them
    

    def __init__(self, model, calibration_set_x, calibration_set_y, alpha,
                call_function_name = None, name = None, kernel = None, verbose = False, raise_error_at_outliers = False):

        if call_function_name != None:
            model.__class__.__call__ = getattr(model.__class__, call_function_name)
        elif hasattr(model, "__call__"): pass
        else: raise ValueError("Couldn't resolve call function")

        self.adaptive = kernel != None
        self.kernel = kernel
        self.verbose = verbose
        self.model = model
        self.alpha = alpha
        self.raise_error_at_outliers = raise_error_at_outliers
        
        self.name = name if name != None else f"{self.__class__.__name__} {model.__class__.__name__}"
        if self.adaptive and name == None: self.name = f"{self.name} LCP, H = {self.kernel.__name__}"
        self.calibrate(calibration_set_x, calibration_set_y)
    


    def _quantile(self, calibration_scores):
        """
        compute the weighted 1-alpha quantile of the scores

        Args:
        ----
            scores: the scores as calculated by the score_distribution

        Returns:
        --------
            q: a function of test points X, that returns a tuple containing:
                q(X)[0]: the empirical 1-alpha quantiles of calibration scores
                         evaluated at each test point in X,
                         as a 1D np.array of length n_test
                q(X)[1]: effective sample sizes, X, N_test x 1 np.array
        """
        if self.adaptive:
            return lambda X: self._weighted_quantile(calibration_scores, X)
        else:
            #the level on which we take the quantile of the scores
            level = np.ceil((self.n_cal+1)*(1-self.alpha))/self.n_cal
            q = np.quantile(calibration_scores, level)
            return lambda X: (np.ones(len(X))*q, np.ones(len(X))*self.n_cal)

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
        n_test = len(X)
        quantiles = []
        effective_sample_sizes = []
        n_kernel_outliers = 0
    
        if self.verbose:  print(f"\nFitting adaptive quantiles - {self.name}")
        if self.verbose:  iterable = tqdm(self.kernel(self.calibration_set_x, X), total = n_test)
        else:             iterable = iter(self.kernel(self.calibration_set_x, X))
        # [H_n+1, i for i in 1...n]
        


        #for each data point, Xi, compute the weighted 1-alpha quantile
        for i, kernel_values_raw in enumerate(iterable):
            #sort kernel values with same mask as calibration scores
            kernel_values = kernel_values_raw[ix]
            kernel_values_cum_sum = np.cumsum(kernel_values)

            #if all kernel values are 0
            if np.sum(kernel_values) == 0 or kernel_values_cum_sum[-1] == 0:

                #create error messages
                msg_txt = f'Encountered test point, X[{i}], where kernel(x) = 0 for all x in calibration set'
                msg_X_i = f'X[{i}] = {X[i]}'
                if self.raise_error_at_outliers:
                    raise ValueError(msg_txt + '\n' + msg_X_i)
                elif self.verbose:
                    print('\n' + msg_txt + '\nkernel(x) was set to 1 for all x in calibration set\n' + msg_X_i)

                #if allowed to keep running - use constant kernel i.e. don't weigh any points.
                kernel_values = np.ones_like(kernel_values_raw)
                kernel_values_cum_sum = np.cumsum(kernel_values)
                effective_sample_sizes.append(None)
                norm_constant = 1
            #otherwize
            else:
                norm_constant = 1/max(kernel_values_raw)*1e2
                kernel_values = kernel_values*norm_constant
                kernel_values_cum_sum = kernel_values_cum_sum*norm_constant
                if np.sum(kernel_values**2) == 0:
                    raise ValueError('Something went completely wrong with that kernel - all weights sum to 0')
                effective_sample_sizes.append(np.sum(kernel_values)**2/np.sum(kernel_values**2))
            
            # cdf = kernel_values_cum_sum/(kernel_values_cum_sum[-1])#
            cdf = kernel_values_cum_sum/(kernel_values_cum_sum[-1]+1*norm_constant)
            
            # WARN: Theoretically should return a quantile of infinity if the 1-alpha quantile is 
            #  higher than any of the calibration scores. Instead sets the quantile to highest known instead.
            index = binary_search(cdf)
            if index == len(cdf):
                n_kernel_outliers += 1
                index -= 1

            quantiles.append(sorted_scores[index])

        if self.verbose: print(f"Encountered level 1-alpha quantile = infinity in {n_kernel_outliers} test data points")

        return np.array(quantiles), np.array(effective_sample_sizes)

    def calibrate(self, calibration_set_x, calibration_set_y):
        """
        Computes the calibration quantile, q_hat, and sets calibration set

        Args:
        -----
            calibration_set_x: Calibration set inputs 
            calibration_set_y: The true labels/values of the calibration set
        """
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y
        self.n_cal = len(calibration_set_x)

        scores = self.score_distribution()
        self.q = self._quantile(scores)

    def evaluate_coverage(self, X, y):
        """
        Evaluate epirical coverage on test data points.
        
        Args:
        ------
            X: the features of the test data points
            
            y: the true labels/values of the test data points
        
        Returns:
        --------
            results: dataclass object containing:

                cp_model_name: str

                mean_effective_sample_size: int

                empirical_coverage: int

                y_preds: the predictions of the underlying model

                pred_sets: np.ndarray

                effective_sample_sizes: np.ndarray, float, shape = (n_test, 1)

                pred_set_sizes: np.ndarray, float, shape = (n_test, 1)

                kernel_error: np.ndarray, bool, shape = (n_test, 1)

                in_pred_set: np.ndarray, bool, shape = (n_test, 1)

        """
        
        # Compute results - These will be included in output 'result' if they don't begin with "_"
        
        # predictions from ml model, prediction sets, and effective sample sizes
        y_preds, pred_sets, effective_sample_sizes = self.predict(X)
        y_preds = y_preds.squeeze()

        # prediction set sizes
        pred_set_sizes = self._pred_set_sizes(pred_sets)

        # boolean array - True if data point is an outlier according to kernel
        # meaning that kernel(x) = 0 for all calibration points
        kernel_errors = effective_sample_sizes == np.array([None]*len(X))

        mean_effective_sample_size = np.mean(effective_sample_sizes[~kernel_errors].astype(float))
        in_pred_set = self._in_pred_set(pred_sets, y)
        empirical_coverage = np.mean(in_pred_set)

        result = CPEvalData(self.name, mean_effective_sample_size, empirical_coverage, y_preds, pred_sets, effective_sample_sizes, pred_set_sizes, kernel_errors, in_pred_set)

        return result

    def __call__(self,X):
       return self.predict(X)


@dataclass(frozen=True)
class CPEvalData:
        cp_model_name: str
        mean_effective_sample_size: int
        empirical_coverage: int
        y_preds: np.ndarray
        pred_sets: np.ndarray
        effective_sample_sizes: np.ndarray
        pred_set_sizes: np.ndarray
        kernel_errors: np.ndarray
        in_pred_set: np.ndarray