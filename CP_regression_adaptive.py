from conformal_base import Conformal
import numpy as np

class CP_regression_adaptive(Conformal):
    def __init__(self, kernel, model, calibration_set_x, calibration_set_y, alpha):
        self.kernel = kernel
        super().__init__(model, calibration_set_x, calibration_set_y, alpha)

    def score_distribution(self, calibration_set_x, calibration_set_y):
        preds = self.model(calibration_set_x)
        scores = calibration_set_y - preds
        return scores
    
    def calibrate(self, calibration_set_x, calibration_set_y):
        """
        Computes the calibration quantile 
        Args:
            calibration_set_x: Calibration set inputs 
            calibration_set_y: The true labels/values of the calibration set

        Returns:
            Nothing. Sets self.q as the estimated 1-alpha quantile of the true distribution of scores
        """
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y

        scores = self.score_distribution(calibration_set_x, calibration_set_y)
        self.q_lower = self._quantile(scores, 1 - self.alpha / 2)
        self.q_upper = self._quantile(scores, self.alpha / 2)
        self.q_max = self._quantile(scores, 0)
    
    def _quantile(self, scores, alpha):
        n = len(scores)
        weights = lambda X: self.kernel(self.calibration_set_x, X)
        return lambda X: self._weighted_percentile(scores, weights(X), alpha)

    def _weighted_percentile(self, scores, weights, alpha):
        ix = np.argsort(scores)
        scores = scores[ix] # sort data
        quantiles = []
        for weights_ in weights[:, ix]: # sort weights
            weights_cum_sum = np.cumsum(weights_)
            cdf = weights_cum_sum/weights_cum_sum[-1]
            for i, (score, prob) in enumerate(zip(scores,cdf)):
                if prob >= 1-alpha:
                    quantiles.append(scores[max(i, 0)])
                    break
        return np.array(quantiles)
    
    def predict(self, X):
        y_pred = self.model.predict(X)[:,None]
        return y_pred + np.array([self.q_lower(X), self.q_upper(X)]).T