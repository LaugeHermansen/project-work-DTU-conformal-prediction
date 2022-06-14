from .Regression_base import RegressionBase
import numpy as np

class RegressionSquaredError(RegressionBase):

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
        scores = (self.calibration_set_y - preds)**2
        return scores
    
    def predict(self, X):
        """
        Compute confidence interval for new data points

        Args:
        -----
            X: The new data points
            
        Returns:
        --------
            y_pred: underlying model prediction or output
            pred_sets: Prediction sets - one for each test point.
            effective_sample_sizes: N_text x 1 np.array (returned from self.q)
        """

        y_pred = self.model(X)[:,None]
        q, effective_sample_sizes = self.q(X)
        sqrt_q = np.sqrt(q)[:,None]
        return y_pred, y_pred + sqrt_q*[-1,1], effective_sample_sizes