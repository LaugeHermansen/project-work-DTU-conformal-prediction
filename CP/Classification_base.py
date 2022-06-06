from .CP_base import Base
import numpy as np

class ClassificationBase(Base):

    def evaluate_coverage(self, X, y):
        """
        Evaluate epirical coverage on test data points.
        
        Args:
        ------
            X: the features of the test data points
            y: the true labels/values of the test data points
        
        Returns:
        --------
            The empirical coverage of the test data points
            The prediction
        """
        preds = self.predict(X)
        empirical_coverage = np.mean(preds[np.arange(len(y)), y])
        return empirical_coverage
