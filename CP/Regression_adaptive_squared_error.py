from Regression_adaptive_base import CP_regression_adaptive_base
import numpy as np


class Regression_adaptive_squared_error(CP_regression_adaptive_base):

    def score_distribution(self, calibration_set_x, calibration_set_y):
        preds = self.model(calibration_set_x)
        scores = (calibration_set_y - preds)**2
        return scores
    
    def predict(self, X):
        y_pred = self.model.predict(X)[:,None]
        sqrt_q = np.sqrt(self.q(X))[:,None]
        return y_pred + sqrt_q*[-1,1]