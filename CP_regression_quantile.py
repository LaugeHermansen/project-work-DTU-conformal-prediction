from conformal_base import Conformal
import numpy as np


class CP_regression_quantile(Conformal):
    def score_distribution(self, calibration_set_x, calibration_set_y):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

        Returns:
            All scores of the labels of the calibration set
        """
        #maybe do this
        #return self.score(calibration_set_x,calibration_set_y)

        preds = self.model(calibration_set_x)
        scores = np.max([preds[:, 0] - calibration_set_y, calibration_set_y - preds[:, -1]], axis=0)
        
        return scores

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
        Returns:
            An interval or set of confidence
        """
        y_pred = self.model.predict(X)

        return y_pred[:, [0, -1]] + (self.q * np.array([-1, 1]))
