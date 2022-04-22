#from audioop import reverse
import numpy as np

class Conformal():
    def __init__(self, model, calibration_set_x, calibration_set_y, alpha):
        self.model = model
        self.alpha = alpha
        self.calibrate(calibration_set_x, calibration_set_y)
        
    def score_distribution(self, calibration_set_x, calibration_set_y):
        """
        Compute the scores of the calibration set.

        Args:
            calibration_set_x: the features of the calibration set
            calibration_set_y: the true labels/values of the calibration set

        Returns:
            All calibration scores of the correct label
        """
        raise NotImplementedError()

    def score(self, outputs):
        """
        Compute score of new data
        Args:
            outputs: The outputs of the model given the new data points
        Returns:
            Scores of the outputs of the new data points
        """
        raise NotImplementedError()

    def _quantile(self, scores, alpha):
        """
        get the empirical 1-alpha quantile of calibration data

        Parameters:
        ----------
            - scores: conformal scores for each data point
            - alpha: significance level of prediction set

        Return:
        -------
            - the empirical 1-alpha quantile of calibration data
        """
        n = len(scores)
        q = np.ceil((n+1)*(1-alpha))/n

        return np.quantile(scores, q)
    

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
            
        Returns:
            An interval or set of confidence
        """
        raise NotImplementedError()

    def calibrate(self, calibration_set_x, calibration_set_y):
        """
        Computes the calibration quantile 
        Args:
            calibration_set_x: Calibration set inputs 
            calibration_set_y: The true labels/values of the calibration set

        Returns:
            Nothing. Sets self.q as the estimated 1-alpha quantile of the true distribution of scores
        """
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y

        scores = self.score_distribution(calibration_set_x, calibration_set_y)
        self.q = self._quantile(scores, self.alpha)

    def __call__(self,X):
       return self.predict(X)


class CP_softmax(Conformal):
    def score_distribution(self, calibration_set_x, calibration_set_y):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

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

class CP_cumulative_softmax(Conformal):
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

class CP_regression(Conformal):
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

        preds = self.model(calibration_set_x)
        scores = np.max([preds[:, 0] - calibration_set_y, calibration_set_y - preds[:, -1]], axis=0)
        
        return scores

    def predict(self, X):
        """
        Compute confidence interval for new data points
        Args:
            X: The new data points
        Returns:
            An interval or set of confidence
        """
        y_pred = self.model.predict(X)

        return y_pred[:, [0, -1]] + (self.q * np.array([-1, 1]))

class CP_regression_adaptive(Conformal):
    def __init__(self, kernel, model, calibration_set_x, calibration_set_y, alpha):
        self.kernel = kernel
        super().__init__(model, calibration_set_x, calibration_set_y, alpha)

    def score_distribution(self, calibration_set_x, calibration_set_y):
        preds = self.model(calibration_set_x)
        scores = calibration_set_y - preds
        return scores
    
    def calibrate(self, calibration_set_x, calibration_set_y):
        """
        Computes the calibration quantile 
        Args:
            calibration_set_x: Calibration set inputs 
            calibration_set_y: The true labels/values of the calibration set

        Returns:
            Nothing. Sets self.q as the estimated 1-alpha quantile of the true distribution of scores
        """
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y

        scores = self.score_distribution(calibration_set_x, calibration_set_y)
        self.q_lower = self._quantile(scores, 1 - self.alpha / 2)
        self.q_upper = self._quantile(scores, self.alpha / 2)
        self.q_max = self._quantile(scores, 0)
    
    def _quantile(self, scores, alpha):
        n = len(scores)
        weights = lambda X: self.kernel(self.calibration_set_x, X)
        return lambda X: self._weighted_percentile(scores, weights(X), alpha)

    def _weighted_percentile(self, scores, weights, alpha):
        ix = np.argsort(scores)
        scores = scores[ix] # sort data
        quantiles = []
        for weights_ in weights[:, ix]: # sort weights
            weights_cum_sum = np.cumsum(weights_)
            cdf = weights_cum_sum/weights_cum_sum[-1]
            for i, (score, prob) in enumerate(zip(scores,cdf)):
                if prob >= 1-alpha:
                    quantiles.append(scores[max(i, 0)])
                    break
        return np.array(quantiles)
    
    def predict(self, X):
        y_pred = self.model.predict(X)[:,None]
        return y_pred + np.array([self.q_lower(X), self.q_upper(X)]).T



