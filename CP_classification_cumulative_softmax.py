from conformal_base import Conformal
import numpy as np

class CP_classification_cumulative_softmax(Conformal):
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

        all_scores = self.model(calibration_set_x)
        indices = np.argsort(-all_scores, axis=1) # The minus is to get the max element in front 
        indices_reverse = np.argsort(indices,  axis=1)
        sorted_scores = np.take_along_axis(all_scores, indices, axis=1)
        true_labels_idx = indices_reverse[np.arange(len(calibration_set_y)), calibration_set_y]

        true_scores = [sum(sorted_score[:true_label+1]) for sorted_score, true_label in zip(sorted_scores, true_labels_idx)]
        return true_scores


    def score(self, model_out, calibration_set_y = None):
        """
        Compute score of new data
        Args:
            model_out: The outputs of the model given the new data points
        Returns:
            Scores of the model_out of the new data points
        """
        indices = np.argsort(-model_out, axis=1) # The minus is to get the max element in front 
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(model_out, indices, axis=1)

        scores = np.cumsum(sorted, axis = 1)
        scores = np.take_along_axis(scores, reverse_indices, axis=1)

        # maybe do this
        # if calibration_set_y != None:
        #     scores = scores[np.arange(len(scores)), calibration_set_y]

        return scores

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
        Returns:
            An interval or set of confidence
        """
        y_pred = self.model(X)
        scores = self.score(y_pred)

        return scores <= self.q

