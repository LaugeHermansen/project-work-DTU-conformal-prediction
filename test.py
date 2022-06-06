import types
from sklearn.datasets import make_regression
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt

from GP.gaussian_process_wrapper import GaussianProcessModelWrapper


class hej:
    def __init__(self, a):
        self.a = a
    
    def test(self, b):
        print(b)


alpha = 0.20
quantiles = [alpha/2, 0.5, 1-alpha/2]
n_features = 1
n_informative = 1
n_targets = 1
noise = 6


X, y = make_regression(n_samples=40, n_features=n_features, n_informative=n_informative, n_targets=n_targets, noise=noise,random_state=1234)
y = y*np.sin(X[:, 0]) + y
# y = y - 100 *(y >= 50)
# y_2 = y + 10
y = (y - np.mean(y)) / np.std(y)

x_grid = np.linspace(np.min(X), np.max(X), num=300).reshape((-1, 1))

# model = GaussianProcessModel(None, X, y, alpha)
# model = GaussianProcessRegressor(Matern(length_scale=0.2, length_scale_bounds="fixed", nu=1), 
model = GaussianProcessRegressor(Matern(), 
                                 alpha=1e-1, 
                                 n_restarts_optimizer=0)
model.fit(X, y)

def predict(model, X, alpha):
    y_mean, y_std = model.predict(X, return_std=True)
    pred_interval = y_mean[:, None] + np.array([-1, 1]) * y_std[:,None] * norm.ppf(q = 1-alpha)
    return y_mean, pred_interval

# y_pred, y_intervals = model.predict(x_grid)
y_pred, y_intervals = predict(model, x_grid, alpha)
_, std = model.predict(x_grid, return_std=True)

plt.plot(X, y, ".")
# plt.ylim((-100, 100))
# plt.show()
plt.plot(x_grid, y_pred)
# plt.plot(x_grid, std)
plt.plot(x_grid, y_intervals[:, 1])
plt.plot(x_grid, y_intervals[:, 0])
plt.show()