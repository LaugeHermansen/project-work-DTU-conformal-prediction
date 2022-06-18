#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from Models import MultipleQuantileRegressor

from CP import RegressionSquaredError, RegressionQuantile, CPEvalData


from Toolbox.plot_helpers import barplot, scatter, compute_lim
from Toolbox.kernels import mahalanobis_exponential, exponential, KNN, mahalanobis_KNN
from Toolbox.tools import get_all_cp_models, multiple_split, evaluate2, GPWrapper, mpath, CPRegressionResults

from protein_data_set import get_protein_data

import os
import random

random.seed(123456)
np.random.seed(123456)

results_path = mpath("results/protein_results/")
pca = PCA()
st = StandardScaler()


X,y,stratify, data = get_protein_data(y_i = 0)

X_transform = X
X_transform = np.hstack((X_transform, X_transform**2))
# X_transform = np.hstack((X_transform, np.sin(X_transform), np.cos(X_transform)))

X_standard = st.fit_transform(X_transform)

(train_X, cal_X, test_X,
train_y, cal_y, test_y,
train_strat, cal_strat, test_strat) = multiple_split((0.1,0.6,0.3), X_standard,y,stratify, keep_frac = 1.)

# scatter(X_standard, y, alpha = 0.6)
# plt.title("$y_{true}$ on standardized X")
# plt.show()



#set significance level of CP
alpha = 0.2
length_scale = 1.0


# Fit Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 100, n_jobs = 6)
rf.fit(train_X, train_y)


# create and fit cpmodels
cpKNN = lambda K, model: RegressionSquaredError(model, cal_X, cal_y, alpha, 'predict', kernel = KNN(K), name=f'RF, NLCP, KNN({K})', verbose = True)

cprf_exp   =   RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', kernel = exponential(length_scale), name=f'RF, NLCP, exp({length_scale})', verbose = True)
cprf_50NN  = cpKNN(50, rf)
# cprf_100NN  = cpKNN(100, rf)
cprf_300NN  = cpKNN(300, rf)
cprf_1000NN  = cpKNN(1000, rf)

cprf_basic =  RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', name = "RF, Basic CP")

#Evaluate CP models
cp_models = get_all_cp_models(locals())
cp_results = [cp.evaluate_coverage(test_X, test_y) for cp in cp_models]




# Save results
experiment_name = "Final_results_2"

path = mpath(results_path + experiment_name)


experiment_data = CPRegressionResults(cp_results, data, X_standard, y, stratify,
                              train_X, cal_X, test_X,
                              train_y, cal_y, test_y,
                              train_strat, cal_strat, test_strat,
                              )

experiment_data.save(path + "CP_data")
