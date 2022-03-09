import numpy as np

class conformal():
    def __init__(self, model, calibration_set_x, calibration_set_y, alpha):
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y
        self.model = model
        self.alpha = alpha

        self.q = self.quantile(alpha)

    def score_distribution(self):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

        Returns:
            All calibration scores of the correct label
        """
        raise NotImplementedError

    def score(self, labels):
        """
        Compute score of new data
        Args:
            labels: The outputs of the model given the new data points
        Returns:
            Scores of the labels of the new data points
        """
        raise NotImplementedError

    def quantile(self, alpha):
        """
        get the empirical 1-alpha quantile of calibration data

        Parameters:
        ----------
            - calibration_dl: torch Dataloader object
            - alpha: significance level of prediction set

        Return:
        -------
            - the empirical 1-alpha quantile of calibration data
        """
        n = len(self.calibration_set_y)
        q = np.ceil((n+1)*(1-alpha))/n
        scores = self.score_distribution()

        return np.quantile(scores, q)

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
        Returns:
            An interval or set of confidence
        """
        raise NotImplementedError


class CP_softmax(conformal):
    def score_distribution(self):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

        Returns:
            All scores of the labels of the calibration set
        """
        all_scores = -self.model(self.calibration_set_x)
        true_scores = all_scores[self.calibration_set_y]
        return true_scores

    def score(self, labels):
        """
        Compute score of new data
        Args:
            labels: The outputs of the model given the new data points
        Returns:
            Scores of the labels of the new data points
        """
        return -labels

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


