from sklearn.covariance import empirical_covariance
from .CP_base import Base
import numpy as np


class ClassificationBase(Base):
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

        # return scores <= self.q

        # new code ----------
        scores_idx = np.argsort(scores, axis = 1)
        ret = np.zeros_like(y_pred).astype(bool)
        for i in range(len(y_pred)):
            for j in scores_idx[i]:
                ret[i,j] = True
                if scores[i,j] >= self.q: break
        return ret
        # -----------------------


    def evaluate_coverage(self, X, y):
        """
        Evaluate epirical coverage on test data points.
        
        Args:
            X: the features of the test data points
            y: the true labels/values of the test data points
        
        Returns:
            The empirical coverage of the test data points
            The prediction
        """
        preds = self.predict(X)
        empirical_coverage = np.mean(preds[np.arange(len(y)), y])
        return empirical_coverage
