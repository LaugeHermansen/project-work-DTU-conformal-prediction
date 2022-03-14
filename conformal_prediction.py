import numpy as np

class conformal():
    def __init__(self, model, calibration_set_x, calibration_set_y, alpha):

        self.model = model
        self.alpha = alpha
        self.calibrate(calibration_set_x, calibration_set_y)


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

    def calibrate(self,calibration_set_x, calibration_set_y):
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y
        self.q = self.quantile(self.alpha)

    def __call__(self,X):
       return self.predict(self,X)


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
        true_scores = all_scores[list(range(len(self.calibration_set_y))), self.calibration_set_y]
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

class CP_cumulative_softmax(conformal):
    def score_distribution(self):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

        Returns:
            All scores of the labels of the calibration set
        """
        all_scores = self.model(self.calibration_set_x)
        indices = np.argsort(all_scores, axis=1)
        sorted_scores = np.take_along_axis(all_scores, indices, axis=1)
        true_labels = indices[list(range(len(self.calibration_set_y))), self.calibration_set_y]

        true_scores = [sum(sorted_score[:true_label]) for sorted_score, true_label in zip(sorted_scores, true_labels)]
        return true_scores

    def score(self, labels):
        """
        Compute score of new data
        Args:
            labels: The outputs of the model given the new data points
        Returns:
            Scores of the labels of the new data points
        """
        indices = np.argsort(labels, axis=1)
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(labels, indices, axis=1)
        scores = np.zeros_like(labels)
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                scores[i, j] = sorted[i, j] + scores[i, j-1] if j != 0 else sorted[i,j]
        scores = np.take_along_axis(scores, reverse_indices, axis=1)
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

