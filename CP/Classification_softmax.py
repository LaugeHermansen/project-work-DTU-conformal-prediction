from .Classification_base import ClassificationBase
# from .CP_base import Base
import numpy as np


class ClassificationSoftmax(ClassificationBase):
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




