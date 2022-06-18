#%%
from dataclasses import dataclass, asdict
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
from Toolbox.tools import get_all_cp_models, multiple_split, evaluate2, GPWrapper

from protein_data_set import get_protein_data

import os

@dataclass
class CPRegressionResults:
    cp_results: CPEvalData
    data: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    stratify: np.ndarray
    train_X: np.ndarray
    cal_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    cal_y: np.ndarray
    test_y: np.ndarray
    train_strat: np.ndarray
    cal_strat: np.ndarray
    test_strat: np.ndarray

    def save(self, path):
        pd.to_pickle(self, path)
    
    def load(path):
        ret = pd.read_pickle(path)
        if ret.__class__ == CPRegressionResults: return ret
        else: raise FileExistsError('File is not a CPRegressionResults object')
     


def finalize_plot(show = False, path = None):
    plt.tight_layout()
    if path != None:
        plt.savefig(path)
    if show: plt.show()
    else: plt.clf()

def mpath(path):
    path = path.strip('/')
    path_list = path.split('/')
    for i, p in enumerate(path_list):
        root = "./" + "/".join(path_list[:i])
        if not p in os.listdir(root):
            os.mkdir(root + "/" + p)
    return path + "/"

results_path = mpath("results/protein_results/")

#%%

y_i = 0
X,y,stratify, data = get_protein_data(y_i = y_i)

# decimal = [1,1,1,3,1,1,1,1,0,1]

# for a,b,d in zip(data.min(),data.max(),decimal):
#     a = np.round(a, d)
#     b = np.round(b, d)
#     if d == 0:
#         a = int(a)
#         b = int(b)
#     print(f"$[{a}~~,~~{b}]$")


pca = PCA()
st = StandardScaler()

X_transform = X
X_transform = np.hstack((X_transform, X_transform**2))
# X_transform = np.hstack((X_transform, np.sin(X_transform), np.cos(X_transform)))

X_standard = st.fit_transform(X_transform)
X_pca = pca.fit_transform(X_standard)

(train_X, cal_X, test_X,
train_y, cal_y, test_y,
train_strat, cal_strat, test_strat) = multiple_split((0.1,0.45,0.45), X_standard,y,stratify, keep_frac = 1.)

scatter(X_pca, y, alpha = 0.6)
plt.title("$y_{true}$ on PCs")
plt.show()
# scatter(X_standard, y, alpha = 0.6)
# plt.title("$y_{true}$ on standardized X")
# plt.show()


#%%

#set significance level of CP
alpha = 0.2
K = 200
length_scale = 1.0

# fit predictive models
#%%
# Fit Random Forest Regressor
rf = RandomForestRegressor(n_estimators = 100, n_jobs = 6)
rf.fit(train_X, train_y)
lr = LinearRegression()
lr.fit(train_X, train_y)
#%%
gp = GPWrapper(Matern() + WhiteKernel(), normalize_y=True)
# gp.fit(train_X, train_y)


#%%

# create and fit cpmodels
cprf_exp       =   RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', kernel = exponential(length_scale), name=f'RF, NLCP, kernel exponential ({length_scale})', verbose = True)
cplr_exp       =   RegressionSquaredError(lr, cal_X, cal_y, alpha, 'predict', kernel = exponential(length_scale), name=f'LR, NLCP, kernel exponential ({length_scale})', verbose = True)
# cpgp_exp       =   RegressionQuantile(gp, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_exponential(length_scale), name=f'GP, NLCP, kernel {K}-NN', verbose = True)
# cprf_KNN_maha    =   RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_KNN(K), name=f'RF, NLCP, Mahalanobis {K}-NN', verbose = True)
cprf_KNN         =   RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', kernel = KNN(K), name=f'RF, NLCP, kernel {K}-NN', verbose = True)
# cplr_KNN_maha    =   RegressionSquaredError(lr, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_KNN(K), name=f'LR, NLCP, Mahalanobis {K}-NN', verbose = True)
cplr_KNN         =   RegressionSquaredError(lr, cal_X, cal_y, alpha, 'predict', kernel = KNN(K), name=f'LR, NLCP, kernel {K}-NN', verbose = True)
# cpgp_KNN       =   RegressionQuantile(gp, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_KNN(K), name=f'GP, NLCP, kernel {K}-NN', verbose = True)
cprf_basic       =   RegressionSquaredError(rf, cal_X, cal_y, alpha, 'predict', name = "RF, Basic CP")
cplr_basic       =   RegressionSquaredError(lr, cal_X, cal_y, alpha, 'predict', name = "LR, Basic CP")
# cpgp_basic     =   RegressionQuantile(gp, cal_X, cal_y, alpha, 'predict', name = "GP Basic CP")


#Evaluate CP models
# cp_models = [cprf_KNN, cprf_basic]
# cp_results = [cp.evaluate_coverage(test_X, test_y) for cp in cp_models]
cp_models = get_all_cp_models(locals())
cp_results = [cp.evaluate_coverage(test_X, test_y) for cp in cp_models]

experiment_name = "ex 2"

path = mpath(results_path + experiment_name)


experiment_data = CPRegressionResults(cp_results, data, X, y, stratify,
                              train_X, cal_X, test_X,
                              train_y, cal_y, test_y,
                              train_strat, cal_strat, test_strat,
                              )

experiment_data.save(path + "CP_data")

#%% --------------------------------------------------------------------------------
# Load experiment data

experiment_name = "ex 2"
path = mpath(results_path + experiment_name)
experiment_data = CPRegressionResults.load(path + "CP_data")

cp_results =     experiment_data.cp_results
data =           experiment_data.data
X =              experiment_data.X
y =              experiment_data.y
stratify =       experiment_data.stratify
train_X =        experiment_data.train_X
cal_X =          experiment_data.cal_X
test_X =         experiment_data.test_X
train_y =        experiment_data.train_y
cal_y =          experiment_data.cal_y
test_y =         experiment_data.test_y
train_strat =    experiment_data.train_strat
cal_strat =      experiment_data.cal_strat
test_strat =     experiment_data.test_strat

cp_result_group_names = set(map(lambda x: x.cp_model_name[:2], cp_results))
cp_result_groups = {group:[result for result in cp_results if result.cp_model_name[:2] == group] for group in cp_result_group_names}

#%%
# plot a bunch of stuff




test_X_pca = pca.transform(test_X)

group_dict = {'LR': 'Linear Regression',
              'RF': 'Random Forest'}

plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"] = 200
for group_name, results in cp_result_groups.items():
    for result in results:
        print("")
        print(result.cp_model_name)
        print("coverage:", result.empirical_coverage)
        print("expected prediction interval size:", np.mean(result.pred_set_sizes))
        print("mean effective sample size", result.mean_effective_sample_size)
        print("kernel outliers:", np.sum(result.kernel_errors))
        print("Model mean squared error:", mean_squared_error(test_y, result.y_preds))
        
        if len(set(result.pred_set_sizes)) > 10:
            plt.hist(result.pred_set_sizes, density = True, label = result.cp_model_name[4:], alpha = 0.4, bins = 40)
        elif "Basic" in result.cp_model_name:
            plt.vlines(np.mean(result.pred_set_sizes), *plt.gca().get_ylim(), label = "Basic CP")
    plt.suptitle(group_dict[group_name])
    plt.legend()
    # plt.tight_layout()
    # plt.show()
    finalize_plot(path = mpath(path + "pred_set_size_histograms") + group_dict[group_name])
    
#%%

n_bins = 20
discard_frac = 0.1

# model_names = [f'NLCP, kernel {K}-NN', "Basic CP"]

for group_name, results in cp_result_groups.items():
    for i in range(2):
        min_x, max_x = np.quantile(test_X_pca[:,i], [discard_frac/2,1-discard_frac/2])
        mask = (min_x <= test_X_pca[:,i]) & (max_x >= test_X_pca[:,i])
        dx = (max_x-min_x)/n_bins
        x_labels = np.linspace(min_x + dx/2, max_x + dx/2, n_bins, endpoint=True)
        array = test_X_pca[mask,i]
        
        fig, ax1 = plt.subplots(dpi = 200, figsize = (6,4))
        ax2 = ax1.twinx()
        max_c =  0
        min_c =  float('inf')

        for j, result in enumerate(results):
            conditional_coverage, count = evaluate2(array, result.in_pred_set, n_bins, min_x, max_x)
            pred_set_size, count = evaluate2(array, result.pred_set_sizes, n_bins, min_x, max_x)

            max_c =  max(max_c, np.max(conditional_coverage))
            min_c =  min(min_c, np.min(conditional_coverage))
            ax1.plot(x_labels, conditional_coverage, label = result.cp_model_name[4:])
        ax2.plot(x_labels, count, label = "Bin Volume", color = 'r')

        ax2.set_ylim(0, np.max(count)*4)
        labels = ax2.get_yticks()
        ax2.set_yticks(labels[0::2]/4)

        ax1.set_ylim(min_c - (max_c-min_c)*0.5, max_c + (max_c-min_c)*0.5)
        plt.suptitle(group_dict[group_name])
        # labels = ax2.get_yticks()
        # ax2.set_yticks(labels/3)


        ax1.set_xlabel(f'PCA {i + 1}')
        ax1.set_ylabel('Empirical bin conditional coverage')
        ax2.set_ylabel('Number of data points')


        ax1.legend(loc = 'upper left')
        ax2.legend()
        
        finalize_plot(path = mpath(path + "binned_FSC") + group_dict[group_name])
        # plt.show()
        
    



#%%

plot_alpha = 0.6
s = 3
lim_level = 0.005

def replace(string, old_list, new):
    for c in old_list:
        string = string.replace(c,new)
    return string

for result in cp_results:
    plt.figure(figsize = (10,10), dpi = 300)
    ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    ax1.plot(
        test_y, 
        result.pred_set_sizes, 
        '.', markersize = s/2,
        alpha = plot_alpha
    )
    ax1.set_title("Prediction set set size vs $y_{true}$")
    ax1.set_xlabel("True label, $y_{true}$")
    ax1.set_ylabel(r"Prediction set size $|\tau(X)|$")
    ax1.set_ylim(compute_lim(result.pred_set_sizes, lim_level))

    ax2 = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    ax2.plot(
        np.abs(test_y-result.y_preds), 
        result.pred_set_sizes, 
        '.', markersize=s/2,
        alpha = plot_alpha
    )
    ax2.set_title("Prediction set size vs true absolute difference")
    ax2.set_xlabel("$|y_{pred}-y_{true}$|")
    ax2.set_ylabel(r"Prediction set size $|\tau(X)|$")
    ax2.set_ylim(compute_lim(result.pred_set_sizes, lim_level))

    # plt.show()
    # continue

    ax3 = plt.subplot2grid((4, 5), (0, 2), rowspan=2, colspan=3)
    quantity = result.pred_set_sizes
    scatter(test_X_pca, quantity, adapt_lim_level=lim_level, alpha=plot_alpha, s=s)
    ax3.set_title('pred set sizes on first two PCs')
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.plot()

    plt.tight_layout()

    ax4 = plt.subplot2grid((4, 5), (2, 2), rowspan=2, colspan=3)
    quantity = np.abs(result.y_preds - test_y)
    scatter(test_X_pca, quantity, adapt_lim_level=lim_level, alpha=plot_alpha, s=s)
    ax4.set_title('Squared error on first two PCs')
    ax4.set_xlabel("PC 1")
    ax4.set_ylabel("PC 2")
    ax4.plot()
    plt.suptitle(group_dict[result.cp_model_name[:2]] + result.cp_model_name[2:], fontsize = 15)

    finalize_plot(path = mpath(path + "overview_plots") + replace(result.cp_model_name, r'.<>:"/\|?*', "_"))






    # Plot adaptive coverage for linear regression
    
    # plt.plot(test_y, pred_intervals_ad[:,1] - pred_intervals_ad[:,0], '.', alpha = 0.05)
    # plt.title("pred set size vs y_true")
    # plt.xlabel("True label, $y_{true}$")
    # plt.ylabel(r"Prediction set size $|\tau(X)|$")
    # plt.show()


    # plt.plot(np.abs(test_y-y_preds_ad), pred_intervals_ad[:,1] - pred_intervals_ad[:,0], '.', alpha = 0.05)
    # plt.title("pred set size vs true absolute difference")
    # plt.xlabel("$|y_{pred}-y_{true}$|")
    # plt.ylabel(r"Prediction set size $|\tau(X)|$")
    # plt.show()


    # test_X_pca = pca.transform(test_X)
    # plt.scatter(test_X_pca[:,0],test_X_pca[:,1],edgecolors='none',s=6, alpha = alpha,c=pred_intervals_ad[:,1] - pred_intervals_ad[:,0])
    # plt.colorbar()
    # plt.title('pred set sizes on first two PCs')
    # plt.xlabel("PC 1")
    # plt.ylabel("PC 2")
    # plt.plot()
    # plt.show()

#%%

# Plot the coverage based on class (y-intervals)

# bar = []
# height_ad = []
# height_st = []
# for y_class in sorted(set(stratify)):
#     bar.append(y_class)
#     height_ad.append(np.mean(in_pred_set_ad[test_stratify == y_class]))
#     height_st.append(np.mean(in_pred_set_st[test_stratify == y_class]))


# # barplot(bar, (height_ad, height_st), ("adaptive", "static"))

# plt.plot(bar, height_st, label = "static")
# plt.plot(bar, height_ad, label = "adaptive")
# print(np.std(height_ad))
# print(np.std(height_st))
# plt.legend()
# plt.title("Coverage vs y-intervals")
# plt.show()
