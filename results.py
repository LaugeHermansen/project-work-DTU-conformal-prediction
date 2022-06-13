#%% Imports 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Models import MultipleQuantileRegressor
from CP import RegressionAdaptiveSquaredError, RegressionAdaptiveQuantile

from Toolbox.tools import multiple_split
from Toolbox.kernels import mahalanobis_sqe, squared_exponential

alpha = 0.05

class LinearRegressionWrapper(LinearRegression):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

#%% Create datasets 
size = 1000

X = np.random.rand(size)

# Homoscedastic square
a = 3
b = -5
c = 3
x1 = 4*(X - 0.5)
noise = 2
y_square = a*x1**2 + b*x1 + c + noise*np.random.randn(size)

# Homoscedastic sine
amp = 5
freq = 2*np.pi
noise = 1
y_sine = amp*np.sin(X*freq) + noise*np.random.randn(size)

# Scaling variance square 
a = 2
b = -5
c = 3
x1 = 4*(X - 0)
noise = 3
y_square_scaling = a*x1**2 + b*x1 + c + x1*noise*np.random.randn(size)

# Scaling variance sine 
amp = 5
freq = 2*np.pi
noise = 3
y_sine_scaling = amp*np.sin(X*freq) + X*noise*np.random.randn(size)

# large then small then large variance 
a = 3
b = -5
c = 3
x1 = 4*(X - 0.5)
noise = 3
y_hard = a*x1**2 + b*x1 + c + x1*noise*np.random.randn(size)

# mega hard
noise = 1
a = 2
b = -3
c = 2
x1 = 2*(X - 0.5)
y_sine_hard = x1*noise *np.random.randn(size) + a*x1**2 + c*(x1 < 0.5) + b*x1 + noise *np.random.randn(size)/4

#%% Show dataset
plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(1, 3, 1)
plt.plot(X, y_square, ',')

plt.subplot(1, 3, 2)
plt.plot(X, y_hard, ',')

plt.subplot(1, 3, 3)
plt.plot(X, y_sine_hard, ',')


plt.suptitle("The three toy data sets used for regression")
plt.tight_layout()
plt.savefig("./dataset")
plt.show()

#%% split into train, calibrate and test 
#X, X_cali, X_test, y_square, y_square_cali, y_square_test, _, _, _ = multiple_split([0.4, 0.3, 0.3], X, y_square, np.ones_like(y_square))
X, X_cali, X_test, y_square, y_square_cali, y_square_test, y_sine, y_sine_cali, y_sine_test, y_square_scaling, y_square_scaling_cali, y_square_scaling_test, y_sine_scaling, y_sine_scaling_cali, y_sine_scaling_test, y_hard, y_hard_cali, y_hard_test, y_sine_hard, y_sine_hard_cali, y_sine_hard_test _,_,_ = multiple_split([0.4, 0.3, 0.3], X, y_square, y_sine, y_square_scaling, y_sine_scaling, y_hard, y_sine_hard, np.ones_like(y_square))
X = X.reshape(-1,1)
X_cali = X_cali.reshape(-1,1)
X_test = X_test.reshape(-1,1)


#%% Choose dataset 
dataset_name = "squared"
#y, y_cali, y_test = y_square, y_square_cali, y_square_test
#y, y_cali, y_test = y_square_scaling, y_square_scaling_cali, y_square_scaling_test
#y, y_cali, y_test = y_sine_scaling, y_sine_scaling_cali, y_sine_scaling_test
y, y_cali, y_test = y_hard, y_hard_cali, y_hard_test

#%% load models 
# linear no feature transform
lm = LinearRegressionWrapper(n_jobs = 6)
lm.fit(X, y)

# squared features linear model 
lm_squared = LinearRegressionWrapper(n_jobs = 6)
lm_squared.fit(np.hstack((X, X**2)), y)

# quantile regression no feature transforms 
qr = MultipleQuantileRegressor(X, y, quantiles = [alpha/2, 0.5, 1-alpha/2])

# quantile regression squared feature transforms 
qr_squared = MultipleQuantileRegressor(np.hstack((X, X**2)), y, quantiles = [alpha/2, 0.5, 1-alpha/2])

models = [lm, lm_squared, qr, qr_squared]
model_names = ["linear", "linear squared", "quantile regression", "quantile regression squared"]

# #%% linear model 
# lm_square = LinearRegression(n_jobs = 6)
# lm_square.fit([X], y_square)

# lm_sine = LinearRegression(n_jobs = 6)
# lm_sine.fit([X], y_sine)

# lm_square_scaling = LinearRegression(n_jobs = 6)
# lm_square_scaling.fit([X], y_square_scaling)

# lm_sine_scaling = LinearRegression(n_jobs = 6)
# lm_sine_scaling.fit([X], y_sine_scaling)

# #%% Squared feature - linear model 
# lm_square_squared = LinearRegression(n_jobs = 6)
# lm_square_squared.fit([X, X**2], y_square)

# lm_sine_squared = LinearRegression(n_jobs = 6)
# lm_sine_squared.fit([X, X**2], y_sine)

# lm_square_scaling_squared = LinearRegression(n_jobs = 6)
# lm_square_scaling_squared.fit([X, X**2], y_square_scaling)

# lm_sine_scaling_squared = LinearRegression(n_jobs = 6)
# lm_sine_scaling_squared.fit([X, X**2], y_sine_scaling)

# #%% Quantile regression 
# qr_square = MultipleQuantileRegressor([X], y_square, quantiles = [alpha/2, 1-alpha/2])

# qr_sine = MultipleQuantileRegressor([X], y_sine, quantiles = [alpha/2, 1-alpha/2])

# qr_square_scaling = MultipleQuantileRegressor([X], y_square_scaling, quantiles = [alpha/2, 1-alpha/2])

# qr_sine_scaling = MultipleQuantileRegressor([X], y_sine_scaling, quantiles = [alpha/2, 1-alpha/2])

# #%% Quantile regression - squared component 
# qr_square_squard = MultipleQuantileRegressor([X, X**2], y_square, quantiles = [alpha/2, 1-alpha/2])

# qr_sine_squard = MultipleQuantileRegressor([X, X**2], y_sine, quantiles = [alpha/2, 1-alpha/2])

# qr_square_scaling_squard = MultipleQuantileRegressor([X, X**2], y_square_scaling, quantiles = [alpha/2, 1-alpha/2])

# qr_sine_scaling_squard = MultipleQuantileRegressor([X, X**2], y_sine_scaling, quantiles = [alpha/2, 1-alpha/2])

#%% CP's 
X_cali_squard = np.hstack((X_cali, X_cali**2))
X_test_squard = np.hstack((X_test, X_test**2))

cp_static_lm = RegressionAdaptiveSquaredError(lm, X_cali, y_cali, alpha,  verbose=True)
cp_static_lm_squared = RegressionAdaptiveSquaredError(lm_squared, X_cali_squard, y_cali, alpha,  verbose=True)

cp_anobis_lm = RegressionAdaptiveSquaredError(lm, X_cali, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
cp_anobis_lm_squared = RegressionAdaptiveSquaredError(lm_squared, X_cali_squard, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)

cp_static_qr = RegressionAdaptiveQuantile(qr, X_cali, y_cali, alpha,  verbose=True)
cp_static_qr_squared = RegressionAdaptiveQuantile(qr_squared, X_cali_squard, y_cali, alpha,  verbose=True)

cp_anobis_qr = RegressionAdaptiveQuantile(qr, X_cali, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
cp_anobis_qr_squared = RegressionAdaptiveQuantile(qr_squared, X_cali_squard, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)

cp_models = [cp_static_lm, cp_anobis_lm, cp_static_qr, cp_anobis_qr]
cp_model_names = ["Linear Regression", "Linear Regression Adaptive", "Quantile Regression", "Quantile Regression Adaptive"]
cp_models_squared = [cp_static_lm_squared, cp_anobis_lm_squared, cp_static_qr_squared, cp_anobis_qr_squared]
#%% evaluate on test set. 
X_grid = np.arange(0, 1, 0.01).reshape(-1, 1)
X_grid_squard = np.hstack((X_grid, X_grid**2))
colors = ["b", "k", "g", "r"]

def plot_results(model, X_grid, caption=None, color=None):
    preds = model(X_grid)
    plt.plot(X_grid, preds[0], color=color)
    plt.plot(X_grid, preds[1][:, 0], color=color)
    plt.plot(X_grid, preds[1][:, 1], color=color)
    plt.title(caption)

def plot_results_2(model, X_grid, caption=None, color=None):
    preds = model(X_grid)
    plt.plot(X_grid[:, 0], preds[0], color=color)
    plt.plot(X_grid[:, 0], preds[1][:, 0], color=color)
    plt.plot(X_grid[:, 0], preds[1][:, 1], color=color)
    plt.title(caption + " Squared Features")


for i in range(len(cp_models)):
    plt.subplot(2,2, i+1)
    plt.plot(X_test, y_test, ',')
    plot_results(cp_models[i], X_grid, caption=cp_model_names[i])

plt.tight_layout()
plt.savefig(f"C:/Users/david/Desktop/imgs for fag/4inOne")
plt.show()

for i in range(len(cp_models)):
    plot_results_2(cp_models_squared[i], X_grid_squard, caption=cp_model_names[i], color=colors[i])

plt.legend(cp_model_names, labelcolor=colors)
plt.plot(X_test, y_test, ',')
plt.savefig(f"C:/Users/david/Desktop/imgs for fag/4inone Squared Features")
plt.show()