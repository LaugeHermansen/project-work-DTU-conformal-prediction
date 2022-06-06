
import numpy as np

import matplotlib.pyplot as plt

import torch as t
import scipy.optimize
import types
from sklearn.datasets import make_regression
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from matplotlib import pyplot as plt


global step
step = 0

## kernel definition
def squared_exponential_kernel(a, b, lengthscale, variance):
    """ GP squared exponential kernel """
    # compute the pairwise distance between all the point
    # sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    sqdist = (a**2).sum(1)[:,None] + (b**2).sum(1) - 2 * a@b.transpose(1,0)
    return variance * np.e**(-.5 * (1/lengthscale**2) * sqdist)
    # return variance * np.exp(-.5 * (1/lengthscale**2) * sqdist)

def matern52(a, b, lengthscale, variance):
    #C_{5/2}(d) = \sigma^2\left(1+\frac{\sqrt{5}d}{\rho}+\frac{5d^2}{3\rho^2}\right)\exp\left(-\frac{\sqrt{5}d}{\rho}\right)
    # compute the pairwise distance between all the point
    # d2 = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    d2 = (a**2).sum(1)[:,None] + (b**2).sum(1) - 2 * a@b.transpose(1,0)
    return variance * (1 + np.sqrt(5.)*d2**(0.5)/lengthscale + 5*d2/(3 * lengthscale**2)) * np.e**(-np.sqrt(5.)*d2**0.5/lengthscale)

kernel = matern52
autoopt = False

def fit_GP(X, y, Xtest, kernel, lengthscale, kernel_variance, noise_variance, period=1):
    ## we should standardize the data
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    K = kernel(X, X, lengthscale, kernel_variance)
    L = np.linalg.cholesky(K + noise_variance * np.eye(len(X)))

    # compute the mean at our test points.
    Ks = kernel(X, Xtest, lengthscale, kernel_variance)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  #
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    # compute the variance at our test points.
    Kss = kernel(Xtest, Xtest, lengthscale, kernel_variance)
    # s2 = np.diag(K_) - np.sum(Lk ** 2, axis=0)
    covariance = Kss - (v.T @ v)
    # s = np.sqrt(s2)
    return mu, covariance

def loglike(X,y,kernel,lengthscale, kernel_variance, noise_variance):
    #this is not a recommended way of calculating the likelihood from an efficiency point of view
    X = t.from_numpy(np.array(X).reshape(-1, 1))
    y = t.from_numpy(np.array(y).reshape(-1, 1))
    K = kernel(X, X, lengthscale, kernel_variance) + noise_variance * t.eye(len(X),dtype=t.float64)
    return -.5 * y.transpose(1,0)@t.inverse(K)@y - 0.5*t.logdet(K)- .5 * len(X) * np.log(2*np.pi)
    #the below would be better from a stability point of view
    #but the autograd for cholesky seems to be broken at least in pytorch 1.6 
    #L = t.cholesky(K)
    #alpha = t.cholesky_solve(t.cholesky_solve(y,L),L.transpose(1,0))
    #return -.5 * y.transpose(1,0)@alpha - .5 * t.sum(t.diag(L)) - .5 * len(X) * np.log(2*np.pi)

## parameters definition
alpha = 0.20
quantiles = [alpha/2, 0.5, 1-alpha/2]
n_features = 1
n_informative = 1
n_targets = 1
noise = 6
X, y = make_regression(n_samples=1000, n_features=n_features, n_informative=n_informative, n_targets=n_targets, noise=noise,random_state=1234)
y = y*np.sin(X[:, 0]) + y
# y = y - 100 *(y >= 50)
# y_2 = y + 10
y = (y - np.mean(y)) / np.std(y)

x_grid = np.linspace(np.min(X), np.max(X), num=300).reshape((-1, 1))

global lengthscale, kernel_variance, noise_var
lengthscale = 1 # determines the lengths of the wiggle
kernel_variance = 1 # scale factor
noise_var = .2
n_test_point = 100
n_samples = 10
Xtrain = X
ytrain = y
Xtest = x_grid

def optim_hp():
    global lengthscale, kernel_variance, noise_var
    kpar = [t.autograd.Variable(t.tensor(i,dtype=t.float64),requires_grad=True) for i in (lengthscale, kernel_variance, noise_var)]
    opt = t.optim.LBFGS(kpar,lr=0.1,line_search_fn='strong_wolfe')
    
    def closure():
        loss = -loglike(Xtrain,ytrain,kernel,t.abs(kpar[0]),t.abs(kpar[1]),t.abs(kpar[2]))
        opt.zero_grad()
        loss.backward()
        return loss
    for i in range(10):
        opt.step(closure)
    lengthscale = np.abs(kpar[0].item())
    kernel_variance = np.abs(kpar[1].item())
    noise_var = np.abs(kpar[2].item())+np.spacing(1)
    #print(loglike(Xtrain,ytrain,kernel,lengthscale, kernel_variance, noise_var))
    #print((lengthscale, kernel_variance, noise_var))


def ll2(x):
    kpar = [t.autograd.Variable(t.tensor(i,dtype=t.float64),requires_grad=True) for i in x]
    loss = -loglike(Xtrain,ytrain,kernel,t.abs(kpar[0]),t.abs(kpar[1]),t.abs(kpar[2]))
    loss.backward()
    g=np.zeros(len(kpar))
    for i,k in enumerate(kpar):
        g[i] = k.grad.item()
    return (loss.item(),g)

def optim_hp2():
    global lengthscale, kernel_variance, noise_var
    ll=lambda kpar: -loglike(Xtrain,ytrain,kernel,np.abs(kpar[0]),np.abs(kpar[1]),np.abs(kpar[2]))[0][0]
    print(np.array((lengthscale, kernel_variance, noise_var)))
    res=scipy.optimize.minimize(ll2,np.array((lengthscale, kernel_variance, noise_var)),jac=True)
    kpar=res.x
    print(res.x)
    lengthscale = np.abs(kpar[0])
    kernel_variance = np.abs(kpar[1])
    noise_var = np.abs(kpar[2])+np.spacing(1)

    
def optim_hp3():
    global lengthscale, kernel_variance, noise_var
    ll=lambda kpar: -loglike(Xtrain,ytrain,kernel,np.abs(kpar[0]),np.abs(kpar[1]),np.abs(kpar[2]))[0][0]
    res=scipy.optimize.minimize(ll,np.array((lengthscale, kernel_variance, noise_var)))
    kpar=res.x
    lengthscale = np.abs(kpar[0])
    kernel_variance = np.abs(kpar[1])
    noise_var = np.abs(kpar[2])+np.spacing(1)





# from GP.gaussian_process_wrapper import GaussianProcessModelWrapper


class hej:
    def __init__(self, a):
        self.a = a
    
    def test(self, b):
        print(b)







# model = GaussianProcessModel(None, X, y, alpha)
# model = GaussianProcessRegressor(Matern(length_scale=0.2, length_scale_bounds="fixed", nu=1), 

optim_hp()  
mu, covariance = fit_GP(Xtrain, ytrain, Xtest, kernel, lengthscale, kernel_variance, noise_var)
std = np.sqrt(np.diag(covariance))


def predict(model, X, alpha):
    y_mean, y_std = model.predict(X, return_std=True)
    pred_interval = y_mean[:, None] + np.array([-1, 1]) * y_std[:,None] * norm.ppf(q = 1-alpha)
    return y_mean, pred_interval

# y_pred, y_intervals = model.predict(x_grid)
y_intervals = pred_interval = mu[:, None] + np.array([-1, 1]) * std[:,None] * norm.ppf(q = 1-alpha)
y_mean = mu



plt.plot(X, y, ".")
# plt.ylim((-100, 100))
# plt.show()
plt.plot(x_grid, y_mean)
# plt.plot(x_grid, std)
plt.plot(x_grid, y_intervals[:, 1])
plt.plot(x_grid, y_intervals[:, 0])
plt.show()