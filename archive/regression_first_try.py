from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from torch import quantile
import conformal_base as CP 
from scipy.stats import norm

class dumb_model: 
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

class dumber_model(LinearRegression):
    def __call__(self,X):
        return self.predict(X)



def coverage(y, lower, upper):
    return np.mean((lower <= y) & (y <= upper))

def test_dumb_model(quantiles, X, y, y_2, x_grid):
    model = dumb_model(X, y, quantiles=quantiles)
    CP_model = CP.CP_regression_base(model, X, y_2, alpha=0.05)

    preds = CP_model.predict(x_grid)
    plt.plot(x_grid, preds[:, 0])
    plt.plot(x_grid, preds[:, -1])

    before_preds = model(x_grid)
    plt.plot(x_grid, before_preds[:, 0])
    plt.plot(x_grid, before_preds[:, -1])

    plt.legend(["cpL", "cpU", "origL", "origU"])

    plt.plot(X, y_2, '.')


    preds_cp = CP_model.predict(X)
    print(coverage(y_2, *(model(X).T[[0, -1]])))
    print(coverage(y_2, preds_cp[:, 0], preds_cp[:, -1]))

    plt.show()

def test_dumber_model(X,y,x_grid,alpha, kernel):

    model = dumber_model()
    model.fit(X,y)
    CP_model = CP.CP_regression_adaptive(kernel, model, X, y, alpha)

    preds = CP_model(x_grid)
    plt.plot(x_grid, model.predict(x_grid))
    plt.plot(x_grid, preds[:, 0], "-o", markersize=2)
    plt.plot(x_grid, preds[:, -1], "-o", markersize=2)
    # plt.plot(x_grid, q)

    plt.legend(["model", "cpL", "cpU"])
    plt.grid()

    plt.plot(X, y, '.')

    preds = CP_model(X)
    print(coverage(y, preds[:, 0], preds[:, -1]))

    plt.show()


def ramp(X, middles, width):
    ret = []
    for middle in middles:
        ret.append(np.max(np.hstack((width - np.abs(X-middle), np.zeros((len(X), 1)))), axis = 1))
    assert np.array(ret).shape == (len(middles), len(X))
    return np.array(ret)

def gauss(X, means, variance):
    return np.array([norm.pdf(X, loc=mean, scale=variance) for mean in means])

alpha = 0.20
quantiles = [alpha/2, 0.5, 1-alpha/2]
n_features = 1
n_informative = 1
n_targets = 1
noise = 6

X, y = make_regression(n_samples=1000, n_features=n_features, n_informative=n_informative, n_targets=n_targets, noise=noise,random_state=1234)
y = y*np.sin(X[:, 0]) + y
y = y - 100 *(y >= 50)
y_2 = y + 10

x_grid = np.linspace(np.min(X), np.max(X), num=300).reshape((-1, 1))


kernel = lambda a,b: ramp(a,b,1.5)
kernel = lambda a,b: gauss(a, b, 0.2)

# test_dumb_model(quantiles, X, y, y_2, x_grid)
test_dumber_model(X, y, x_grid, alpha, kernel)


# for quantile in quantiles:
#     qmodel = QuantileRegressor(quantile=quantile, alpha=0, solver="highs-ds")
#     qmodel.fit(X, y) 
#     preds = qmodel.predict(x_grid)

#     plt.plot(x_grid, preds)

#plt.plot(X, y, '.')
#plt.show()




