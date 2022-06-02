from conformal_base import Conformal
import numpy as np


class CP_classification_softmax(Conformal):
    def score_distribution(self, calibration_set_x, calibration_set_y):
        """
        Compute the scores of the calibration set.
        
        Args:
            calibration_set_x: the features of the calibration set
            calibration_set_y: the true labels/values of the calibration set

        Returns:
            All scores of the labels of the calibration set
        """
        all_scores = -self.model(calibration_set_x)
        true_scores = all_scores[np.arange(len(calibration_set_y)), calibration_set_y] 
        # As all_scores is a (N x Classes) we need to index each datapoint seperatly.
        return true_scores

    def score(self, model_out):
        """
        Compute score of new data
        Args:
            model_out: The outputs of the model given the new data points
        Returns:
            Scores of the model_out of the new data points
        """
        return -model_out

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
        Returns:
            boolean array NxM where (i,j) = True means that yj in Tau(Xi).
        """
        y_pred = self.model(X)
        scores = self.score(y_pred)

        return scores <= self.q





