from sklearn.linear_model import QuantileRegressor
import numpy as np

class MultipleQuantileRegressor: 
    def __init__(self, X_train, y_train, quantiles=[0.05/2, 1-0.05/2]):
        self.models = []
        for quantile in quantiles:
            qmodel = QuantileRegressor(quantile=quantile, alpha=0, solver="highs-ds")
            qmodel.fit(X_train, y_train)
            self.models.append(qmodel) 
    
    def predict(self, X): 
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return np.array(preds).T
    
    def __call__(self, X):
        return self.predict(X)