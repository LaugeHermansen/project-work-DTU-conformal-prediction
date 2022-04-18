from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import conformal_prediction as CP 

class dumb_model: 
    def __init__(self, X_train, y_train, quantiles=[0.05/2, 1-0.05/2]):
        self.models = []
        for quantile in quantiles:
            qmodel = QuantileRegressor(quantile=quantile, alpha=0, solver="highs-ds")
            qmodel.fit(X, y)
            self.models.append(qmodel) 
    
    def predict(self, X): 
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return preds 
        

alpha = 0.05 
quantiles = [alpha/2, 0.5, 1-alpha/2]
n_features = 1
n_informative = 1
n_targets = 1
noise = 6

X, y = make_regression(n_samples=10000, n_features=n_features, n_informative=n_informative, n_targets=n_targets, noise=noise)
y = y*np.sin(X[:, 0]/3) + y
y_2 = y + 5 
# X = np.random.randn((1000, 100))

x_grid = np.linspace(np.min(X), np.max(X)).reshape((-1, 1))

# for quantile in quantiles:
#     qmodel = QuantileRegressor(quantile=quantile, alpha=0, solver="highs-ds")
#     qmodel.fit(X, y) 
#     preds = qmodel.predict(x_grid)

#     plt.plot(x_grid, preds)

#plt.plot(X, y, '.')
#plt.show()
model = dumb_model(X, y)
CP_model = CP.CP_regression(model, X, y_2, alpha=0.05)

preds = CP_model.predict(x_grid)
plt.plot(x_grid, preds[0])
plt.plot(x_grid, preds[1])

before_lower_pred = CP_model.model[0].predict(x_grid)
before_high_pred = CP_model.model[1].predict(x_grid)
plt.plot(x_grid, before_lower_pred)
plt.plot(x_grid, before_high_pred)

plt.plot(X, y, '.')
plt.show()

