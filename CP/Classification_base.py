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

        return scores <= self.q


    def evaluate_coverage(self, X, y):
        """
        Evaluate the empirical coverage of the test data points.
        
        Args:
            X: the features of the test data points
            y: the true labels/values of the test data points
        
        Returns:
            The empirical coverage of the test data points
        """
        preds = self.predict(X)
        return np.mean(preds[np.arange(len(y)), y])