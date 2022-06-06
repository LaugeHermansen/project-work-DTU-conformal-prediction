from .Regression_adaptive_base import RegressionAdaptiveBase
import numpy as np

class RegressionQuantile(RegressionAdaptiveBase):
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

        preds = self.model(self.calibration_set_x)
        scores = np.max([preds[:, 0] - self.calibration_set_y, self.calibration_set_y - preds[:, -1]], axis=0)
        
        return scores

    def predict(self, X):
        """
        Compute confidence interval for new data points

        Args:
        -----
            X: The new data points
            
        Returns:
        --------
            y_pred: the outputs of the regressor
            pred_interval: the prediction intervals (N_test x 2) matrix
        """
        y_pred = self.model(X)

        return y_pred, y_pred[:, [0, -1]] + (self.q(X) * np.array([-1, 1]))
