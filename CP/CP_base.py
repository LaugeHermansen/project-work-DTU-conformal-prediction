import numpy as np

class Base():
    def __init__(self, model, calibration_set_x, calibration_set_y, alpha):
        self.model = model
        self.alpha = alpha
        self.calibrate(calibration_set_x, calibration_set_y)
        
    def score_distribution(self, calibration_set_x, calibration_set_y):
        """
        Compute the scores of the calibration set.

        Args:
            Takes nothing as the calibration set is in the init

        Returns:
            All scores of the labels of the calibration set
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

    def _quantile(self, scores):
        """
        get the empirical 1-alpha quantile of calibration data

        Parameters:
        ----------
            - scores: conformal scores for each data point

        Return:
        -------
            - the empirical 1-alpha quantile of calibration data
        """
        #number of calibration points
        n = len(scores)

        #the level on which we take the quantile of the scores
        level = np.ceil((n+1)*(1-self.alpha))/n

        return np.quantile(scores, level)
    

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
            Nothing.
            Sets self.q as the estimated 1-alpha quantile of the true distribution of scores
            sets calibration set
        """
        self.calibration_set_x = calibration_set_x
        self.calibration_set_y = calibration_set_y

        scores = self.score_distribution(calibration_set_x, calibration_set_y)
        self.q = self._quantile(scores)
    
    def evaluate_coverage(self, X, y):
        """
        Evaluate the empirical coverage of the test data points.
        
        Args:
            X: the features of the test data points
            y: the true labels/values of the test data points
        
        Returns:
            The empirical coverage of the test data points
        """
        raise NotImplementedError()

    def __call__(self,X):
       return self.predict(X)


