from .Classification_base import ClassificationBase
# from .CP_base import Base
import numpy as np


class ClassificationSoftmax(ClassificationBase):
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
        all_scores = -self.model(self.calibration_set_x)
        true_scores = all_scores[np.arange(self.n_cal), self.calibration_set_y] 
        # As all_scores is a (N x Classes) we need to index each datapoint seperatly.
        return true_scores

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
        return -self.model(X)
    
    def predict(self, X):
        """
        compute prediction set
        
        Args:
        -----
            X: NxM matrix with test points
        
        Returns:
        --------
            pred_set: boolean array Nxc, where c is number of classes, such that pred_set[i,j] = True iff y_j \in Tau(X_i)
        """
        scores = self.score(X)
        return scores <= self.q






