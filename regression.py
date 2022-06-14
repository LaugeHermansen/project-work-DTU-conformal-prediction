#%%
from ast import Str
from collections import defaultdict, namedtuple
import sys, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from Models import MultipleQuantileRegressor

from CP import RegressionSquaredError, RegressionQuantile
import CP
from GP.gaussian_process_wrapper import GaussianProcessModelWrapper


from Toolbox.plot_helpers import barplot, scatter_y_on_pcs
from Toolbox.kernels import mahalanobis_exponential, exponential, KNN
from Toolbox.tools import get_all_cp_models, multiple_split, stratified_coverage

from protein_data_set import X, y, stratify


pca = PCA()
st = StandardScaler()

X_transform = np.hstack((X, X**2, X**3))
X_standard = st.fit_transform(X_transform)
X_pca = pca.fit_transform(X_standard)

train_X, cal_X, test_X, train_y, cal_y, test_y, train_strat, cal_strat, test_strat = multiple_split((0.2,0.3,0.5), X_standard,y,stratify, keep_frac = 0.3)

scatter_y_on_pcs(X_pca, y)

#%%

#set significance level of CP
alpha = 0.2

# fit predictive models

# Fit linear regression model
lm = LinearRegression(n_jobs = 6)
lm.fit(train_X, train_y)


# Fit Random Forest Regressor
# rf = RandomForestRegressor(n_estimators = 100, n_jobs = 6)
# rf.fit(train_X, train_y)

# fit quanilt regressor (wrapper)
qr = MultipleQuantileRegressor(train_X, train_y, quantiles = [alpha/2, 0.5, 1-alpha/2])

# Fit a GP model
# gp_model = GaussianProcessModelWrapper(None, cal_X, cal_y, alpha)


# Fit a GP-CP model
# gp = GaussianProcessRegressor(Matern(), n_restarts_optimizer=10, random_state=0)
# gp.fit(train_X, train_y)
# cpgp_ad = RegressionAdaptiveSquaredError(gp, cal_X, cal_y, alpha, 'predict', kernel = squared_exponential, verbose = True)
# cpgp_st = RegressionAdaptiveSquaredError(gp, cal_X, cal_y, alpha, 'predict')

#%%

# create and fit cpmodels

# cplm_ad_maha = RegressionSquaredError(lm, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_sqe(1), verbose = True)
# cplm_ad_KNN = RegressionSquaredError(lm, cal_X, cal_y, alpha, 'predict', kernel = KNN(20), verbose = True)
cplm_ad_sqe = RegressionSquaredError(lm, cal_X, cal_y, alpha, 'predict', kernel = exponential(1), verbose = True)
# cplm_st = RegressionSquaredError(lm, cal_X, cal_y, alpha, 'predict')
cpqr_ad_maha = RegressionQuantile(qr, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_exponential(1), verbose = True)
# cpqr_ad_sqe = RegressionQuantile(qr, cal_X, cal_y, alpha, 'predict', kernel = squared_exponential(1), verbose = True)
cpqr_st = RegressionQuantile(qr, cal_X, cal_y, alpha, 'predict')

#%%


# Evaluate cp models

# cp_models = [cplm_ad, cplm_st, cpgp_ad, cpgp_st, gp_model]
# cp_models = [cplm_ad_maha, cplm_ad_sqe, cplm_ad_KNN, cplm_st, cpqr_ad_maha, cpqr_ad_sqe, cpqr_st]

# automatically recognize all cp models in the script - put in a list
# clsmembers = set(tuple(zip(*inspect.getmembers(CP, inspect.isclass)))[0])
# cp_models = [var for var, name in vars().items() if var.__class__.__name__ in clsmembers]

cp_models = get_all_cp_models(locals())
# variables = vars()
# cp_models = [variables[var] for var in clsmembers]

# evaluate cp models in cp_models
# Result = namedtuple("Result", ["cp_model", "y_pred", "y_pred_intervals", "y_pred_predicate", "empirical_coverage", "effective_sample_sizes"])
cp_results = [cp.evaluate_coverage(test_X, test_y) for cp in cp_models]


#%%
# plot a bunch of stuff

test_X_pca = pca.transform(test_X)

plt.rcParams["figure.figsize"] = (12, 8)

for result in cp_results:
    print("")
    print(result.cp_model.name)
    print("coverage:", result.empirical_coverage)
    print("expected prediction interval size:", np.mean(result.pred_set_sizes))
    print("mean effective sample size", result.mean_effective_sample_size)
    print("kernel outliers:", np.sum(result.kernel_outlier))
    
    
labels, empirical_conditional_coverage, n_conditional_samples = zip(*map(lambda result: stratified_coverage(test_X[:,3], result.in_pred_set, 20), cp_results))

names = [r.cp_model.name for r in cp_results]

fig, ax = barplot(labels[0], empirical_conditional_coverage,
names, (14,5))

plt.xticks(rotation = 45, fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel("Classes", fontsize = 23)
plt.legend(fontsize = 14, loc='center left', bbox_to_anchor=(1,0.5))
fig.suptitle("Empirical coverage distributed over true labels", fontsize = 30)
fig.tight_layout()
plt.show()

#%%


def stratified_coverage(feature_samples: np.ndarray, in_pred_set: np.ndarray, n_features: int = None):

    """
    Compute either FSC or MSC, depending on the input, `feature_samples`
    It returns all sc's and not just the minimum.

    Args:
    -----
        feature_samples:
            the feature to stratify over.
            for MSC, pass `feature_samples=pred_set_sizes`
            1d boolean array of length N_test 

        in_pred_set:
            the output from `cp_model.evaluate_coverage.in_pred_set`
        
        n_features:
            the number of features in which to partition the 
            data if `feature_samples` is continuous.
            if `None`, the unique values in `feature_samples`
            will be used

    Returns:
    -------
        labels:
            a list of unique labels, 1d array of length |N_labels|

        empirical_conditional_coverage:
            the empicial coverages conditioned on 
            the labels in `labels`.

        n_conditional_samples:
            the number of samples in test set when conditioned
            on the labels in `labels`.
    """


#%%

feature_samples, in_pred_set, n_features = test_X[:,3], cp_results[0].in_pred_set, 20

temp_feature_coverage = defaultdict(list)

if n_features != None:
    label_lookup = {}
    classes_size = len(feature_samples)/n_features
    label_samples = np.zeros(len(feature_samples)).astype(str)
    mask = np.argsort(feature_samples)
    for label in range(n_features):
        start = np.ceil(classes_size*label).astype(int)
        stop = np.ceil(classes_size*(label+1)).astype(int)
        label_samples[mask[start:stop]] = str(label)
        label_lookup[str(label)] = f"({feature_samples[mask[start]]:.3} ,  {feature_samples[mask[min(stop, len(mask)-1)]]:.3}("
else:
    label_samples = feature_samples.copy().astype(str)
    label_lookup = {i:i for i in set(label_samples)}

for label, ips in zip(label_samples, in_pred_set):
    temp_feature_coverage[label].append(ips)

labels = np.zeros(len(temp_feature_coverage)).astype(str)
empirical_conditional_coverage = np.zeros(len(labels))
n_conditional_samples = np.zeros(len(labels)).astype(int)

for i, (label, ips_list) in enumerate(temp_feature_coverage.items()):
    labels[i] = str(label_lookup[label])
    empirical_conditional_coverage[i] = np.mean(ips_list)
    n_conditional_samples[i] = len(ips_list)

print(labels, empirical_conditional_coverage, n_conditional_samples, sep = "\n")







#%%



    # ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    # ax1.plot(
    #     test_y, 
    #     result.pred_set_sizes, 
    #     '.', 
    #     alpha = 0.05
    # )
    # ax1.set_title("pred set size vs y_true")
    # ax1.set_xlabel("True label, $y_{true}$")
    # ax1.set_ylabel(r"Prediction set size $|\tau(X)|$")


    # ax2 = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    # ax2.plot(
    #     np.abs(test_y-result.y_preds), 
    #     result.pred_set_sizes, 
    #     '.', 
    #     alpha = 0.05
    # )
    # ax2.set_title("pred set size vs true absolute difference")
    # ax2.set_xlabel("$|y_{pred}-y_{true}$|")
    # ax2.set_ylabel(r"Prediction set size $|\tau(X)|$")


    # ax3 = plt.subplot2grid((4, 5), (0, 2), rowspan=4, colspan=3)
    # plt.scatter(
    #     test_X_pca[:,0],
    #     test_X_pca[:,1],
    #     edgecolors='none',
    #     s=6, 
    #     alpha = alpha,
    #     c=result.pred_set_sizes
    # )
    # plt.colorbar()
    # ax3.set_title('pred set sizes on first two PCs')
    # ax3.set_xlabel("PC 1")
    # ax3.set_ylabel("PC 2")
    # ax3.plot()
    # plt.suptitle(result.cp_model.name, fontsize = 15)
    # plt.tight_layout()
    # plt.show()
    
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
