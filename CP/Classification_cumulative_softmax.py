from .Classification_base import ClassificationBase
import numpy as np
from Toolbox.plot_helpers import compute_barplot_data

class ClassificationCumulativeSoftmax(ClassificationBase):

    def predict(self, X):
        """
        compute prediction set

        Args:
        -----
            X: NxM matrix with test points

        Returns:
        ---------
            pred_set: boolean array Nxc, where c is number of classes, such that pred_set[i,j] = True iff y_j in Tau(X_i)
        """
        scores = self.score(X)
        pred_set_test = scores <= self.q

        scores_idx = np.argsort(scores, axis = 1)
        pred_set = np.zeros_like(scores).astype(bool)
        for i in range(len(scores)):
            for j in scores_idx[i]:
                pred_set[i,j] = True
                if scores[i,j] >= self.q: break

        diff = np.sum(pred_set, axis = 1) - np.sum(pred_set_test, axis = 1)

        data = compute_barplot_data(diff)

        return pred_set

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
        #compute scores, make sorting mask, and sort.
        all_scores = self.model(self.calibration_set_x)
        indices = np.argsort(-all_scores, axis=1) # The minus is to get the max element in front
        sorted_all_scores = np.take_along_axis(all_scores, indices, axis=1)

        # use indices to get positions of the true lables in the sorted scores and calculate the scores
        true_label_indices = np.argsort(indices,  axis=1)[np.arange(self.n_cal), self.calibration_set_y]
        true_label_scores = [sum(s[:i+1]) for s, i in zip(sorted_all_scores, true_label_indices)]
        return true_label_scores


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
        model_out = self.model(X)
        indices = np.argsort(-model_out, axis=1) # The minus is to get the max element in front 
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(model_out, indices, axis=1)

        scores = np.cumsum(sorted, axis = 1)
        scores = np.take_along_axis(scores, reverse_indices, axis=1)

        return scores
