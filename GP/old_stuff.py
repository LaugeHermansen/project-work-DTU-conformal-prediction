import torch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from itertools import product

# class GaussianProcess:


%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure('GP regression');
ax = fig.add_subplot(111);
global step
step = 0
import torch as t
import scipy.optimize

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

global lengthscale, kernel_variance, noise_var
## parameters definition
lengthscale = 1 # determines the lengths of the wiggle
kernel_variance = 1 # scale factor
noise_var = .2
n_test_point = 100
n_samples = 10
Xtrain = []
ytrain = []
Xtest = np.linspace(-5, 5, n_test_point).reshape(-1,1)

optim_hp()
mu, covariance = fit_GP(Xtrain, ytrain, Xtest, kernel, lengthscale, kernel_variance, noise_var)

def on_sample(event):
    if event.key=='c':
        global step, lengthscale, kernel_variance, noise_var
        ## we want the mean + the std deviation but also some samples from the posterior
        # clear frame
        plt.clf()
        # we have to refit the GP
        mu, covariance = fit_GP(Xtrain, ytrain, Xtest, kernel, lengthscale, kernel_variance, noise_var)
        # we should get the var
        var = np.sqrt(np.diag(covariance))
        # and we have to sample for it
        samples = np.random.multivariate_normal(mu.reshape(-1), covariance, n_samples)  # SxM
        plt.plot(Xtrain, ytrain, 'ro')
        plt.gca().fill_between(Xtest.flat, mu - 3 * var, mu + 3 * var, color='lightblue', alpha=0.5)
        plt.plot(Xtest, mu, 'red')
        for sample_id in range(n_samples):
            plt.plot(Xtest, samples[sample_id])
        plt.axis([-5, 5, -5, 5])
        plt.draw()  # redraw
    elif event.key=='x':
        optim_hp2()
    elif event.key=='v':
        optim_hp3()
    elif event.key=='o':
        optim_hp()

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
    #redraw()
    widg1.children[0].value=kernel_variance
    widg2.children[0].value=lengthscale
    widg3.children[0].value=noise_var

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
    #redraw()
    widg1.children[0].value=kernel_variance
    widg2.children[0].value=lengthscale
    widg3.children[0].value=noise_var
    
def optim_hp3():
    global lengthscale, kernel_variance, noise_var
    ll=lambda kpar: -loglike(Xtrain,ytrain,kernel,np.abs(kpar[0]),np.abs(kpar[1]),np.abs(kpar[2]))[0][0]
    res=scipy.optimize.minimize(ll,np.array((lengthscale, kernel_variance, noise_var)))
    kpar=res.x
    lengthscale = np.abs(kpar[0])
    kernel_variance = np.abs(kpar[1])
    noise_var = np.abs(kpar[2])+np.spacing(1)
    #redraw()
    widg1.children[0].value=kernel_variance
    widg2.children[0].value=lengthscale
    widg3.children[0].value=noise_var
    
mu, covariance = fit_GP(Xtrain, ytrain, Xtest, kernel, lengthscale, kernel_variance, noise_var)
var = np.sqrt(np.diag(covariance))

plt.plot(Xtrain, ytrain, 'ro');
plt.gca().fill_between(Xtest.flat, mu - 3 * var, mu + 3 * var,  color='lightblue', alpha=0.5);
plt.plot(Xtest, mu, 'blue')
plt.axis([-5, 5, -5, 5])
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import FloatSlider
from ipywidgets import Dropdown
from ipywidgets import Checkbox
from IPython.display import display
sliders=[FloatSlider(value=1.,min=0.001, max=10., step=.01,continuous_update=False) for i in range(3)]
sliders[2].value=0.1
kernsel=Dropdown(options=[('squared exponential',squared_exponential_kernel),('matern 5/2',matern52)],value=squared_exponential_kernel)
auto_opt_sel=Checkbox(value=False,description='Auto optimize kernel parameters')

def update1(sigma):
    global lengthscale, kernel_variance, noise_var
    kernel_variance=sigma
    redraw()
    
def update2(l):
    global lengthscale, kernel_variance, noise_var
    lengthscale=l
    redraw()
    
def update3(sigma_n):
    global lengthscale, kernel_variance, noise_var
    noise_var=sigma_n
    redraw()
def update4(Kernel):
    global kernel
    kernel = Kernel
    if autoopt:
        optim_hp()
    redraw()
def update5(auto_opt):
    global autoopt
    autoopt = auto_opt
    if autoopt:
        optim_hp()
    redraw()
    
widg1=interactive(update1,sigma=sliders[0])
widg2=interactive(update2,l=sliders[1])
widg3=interactive(update3,sigma_n=sliders[2])
widg4=interactive(update4,Kernel=kernsel)
widg5=interactive(update5,auto_opt=auto_opt_sel)

fig.canvas.mpl_connect('button_press_event',on_click1)
fig.canvas.mpl_connect('key_press_event',on_sample)
display(widg1);
display(widg2);
display(widg3);
display(widg4);
display(widg5);