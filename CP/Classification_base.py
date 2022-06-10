from .CP_base import Base
import numpy as np

class ClassificationBase(Base):

    def _in_pred_set(prediction_sets, y):
        """
        prediction_sets should be Ntest x C boolean np.arrays
        """
        return prediction_sets[np.arange(len(y)), y]

    # def evaluate_coverage(self, X, y):
    #     """
    #     Evaluate epirical coverage on test data points.
        
    #     Args:
    #     ------
    #         X: the features of the test data points
    #         y: the true labels/values of the test data points
        
    #     Returns:
    #     --------
    #         The empirical coverage of the test data points
    #         The prediction
    #     """
    #     preds = self.predict(X)
    #     empirical_coverage = np.mean(preds[np.arange(len(y)), y])
    #     return empirical_coverage
