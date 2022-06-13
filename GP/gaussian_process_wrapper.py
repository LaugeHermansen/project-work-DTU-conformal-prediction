
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm
import numpy as np

from CP import Base


class GaussianProcessModelWrapper(Base):
    def __init__(self, model, calibration_set_x, calibration_set_y, alpha, call_function_name = None):
        """
        Model and call_function_name are not used in this class.
        """
        # model = GaussianProcessRegressor(Matern(length_scale=0.05, length_scale_bounds="fixed", nu=3), alpha=50, n_restarts_optimizer=0)
        model = GaussianProcessRegressor(Matern() + WhiteKernel())
        
        super().__init__(
            model, 
            calibration_set_x, 
            calibration_set_y, 
            alpha, 
            "predict"
        )


    def predict(self, X):
        """
        Compute confidence interval for new data points

        Args:
        -----
            X: The new data points
            
        Returns:
        --------
            Prediction, and an interval or set of confidence
        """
        y_mean, y_std = self.model.predict(X, return_std=True)
        pred_interval = y_mean[:, None] + np.array([0, -1, 1]) * y_std[:,None] * norm.ppf(q = 1-self.alpha)
        #pred_interval = y_mean[:, None] + np.array([ -1, 1]) * y_std[:,None] * norm.ppf(q = 1-self.alpha)
        #return y_mean, pred_interval
        return pred_interval
        
        

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

        self.model.fit(calibration_set_x, calibration_set_y)

