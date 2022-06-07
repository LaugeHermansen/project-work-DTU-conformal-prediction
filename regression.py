#%%
from collections import namedtuple
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

from CP import RegressionAdaptiveSquaredError
from CP.Regression_quantile import RegressionQuantile
from GP.gaussian_process_wrapper import GaussianProcessModelWrapper


from sklearn.model_selection import train_test_split
from Toolbox.plot_helpers import barplot
from Toolbox.kernels import mahalanobis_sqe, squared_exponential
from Toolbox.tools import multiple_split
#%%

#Read data

data = pd.read_csv('data/CASP.csv', header = 0)

# test for missing data
for c in data.columns:
    print(f"{c}: nans: {np.sum(np.isnan(data[c]))}, {list(set(data[c].apply(type)))}, [{min(data[c])}, {max(data[c])}]")

#%%
#split x and y, and standardize x

X = data[data.columns[1:]].to_numpy()

y = data[data.columns[0]].to_numpy().squeeze()

st = StandardScaler()
X_OG_standard = st.fit_transform(X)
X_standard = np.hstack((X_OG_standard, X_OG_standard**2))#, X_OG_standard**3, X_OG_standard**4))
X_standard = st.fit_transform(X_standard)


#%%

# create pseudo classes, by binning y to be used for stratification

n_classes = 100
classes_size = len(y)/n_classes

stratify = np.empty_like(y)
mask = np.argsort(y)

for i in range(n_classes):
    start = np.ceil(classes_size*i).astype(int)
    stop = np.ceil(classes_size*(i+1)).astype(int)
    stratify[mask[start:stop]] = i
    plt.hist(y[mask[start:stop]], density=True, bins = 1)
plt.show()

# split dataset

train_X, test_X, cal_X, train_y, test_y, cal_y, train_strat, test_strat, cal_strat = multiple_split((0.2,0.3,0.5), X_standard,y,stratify, keep_frac = 0.8)


# train_X, temp_X, train_y, temp_y, train_stratify, temp_stratify = train_test_split(X_standard, y, stratify, test_size=0.9, stratify=stratify, shuffle = True)
# cal_X, test_X, cal_y, test_y, cal_stratify, test_stratify = train_test_split(temp_X, temp_y, temp_stratify, test_size=0.7, stratify=temp_stratify, shuffle = True)

# Run PCA

pca = PCA()

pca.fit(X_standard)

X_pca = pca.transform(X_standard)

plt.scatter(X_pca[:,0],X_pca[:,1],edgecolors='none',s=3, alpha = 0.1,c=y)
plt.colorbar()
plt.title('y on first two PCs')
plt.show()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("explained variance of PCs")
plt.show()
#%%

alpha = 0.2

# Fit simple model (linear regression)
lm = LinearRegression(n_jobs = 6)
# lm = RandomForestRegressor(n_estimators = 100, n_jobs = 6)
lm.fit(train_X, train_y)

#%%

qr = MultipleQuantileRegressor(train_X, train_y, quantiles = [alpha/2, 1-alpha/2])

#%%

# I changed the adaptive regression base a bit
# Now, if you don't specify a kernel, it is just
# the standard CP, but if you specify a kernel,
# it is adaptive

cplm_ad_maha = RegressionAdaptiveSquaredError(lm, cal_X, cal_y, alpha, 'predict', kernel = mahalanobis_sqe(1), verbose = True)
cplm_ad_sqe = RegressionAdaptiveSquaredError(lm, cal_X, cal_y, alpha, 'predict', kernel = squared_exponential(1), verbose = True)
cplm_st = RegressionAdaptiveSquaredError(lm, cal_X, cal_y, alpha, 'predict')
cpqr_st = RegressionQuantile(qr, cal_X, cal_y, alpha, 'predict')

# Fit a GP model
# gp_model = GaussianProcessModelWrapper(None, cal_X, cal_y, alpha)


# Fit a GP-CP model
# gp = GaussianProcessRegressor(Matern(), n_restarts_optimizer=10, random_state=0)
# gp.fit(train_X, train_y)
# cpgp_ad = RegressionAdaptiveSquaredError(gp, cal_X, cal_y, alpha, 'predict', kernel = squared_exponential, verbose = True)
# cpgp_st = RegressionAdaptiveSquaredError(gp, cal_X, cal_y, alpha, 'predict')


# Evaluate
# cp_models = [cplm_ad, cplm_st, cpgp_ad, cpgp_st, gp_model]
cp_models = [cplm_ad_maha, cplm_ad_sqe, cplm_st, cpqr_st]

Result = namedtuple("Result", ["cp_model", "y_pred", "y_pred_intervals", "y_pred_predicate", "empirical_coverage"])
cp_results = [Result(cp, *cp.evaluate_coverage(test_X, test_y)) for cp in cp_models]


#%%
# plot a bunch of stuff

test_X_pca = pca.transform(test_X)

plt.rcParams["figure.figsize"] = (12, 8)

for result in cp_results:
    y_pred_interval_sizes = result.y_pred_intervals[:,1] - result.y_pred_intervals[:,0]
    print("coverage:", result.empirical_coverage)
    print("expected prediction interval size:", np.mean(y_pred_interval_sizes))
    
    
    
    ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    ax1.plot(
        test_y, 
        y_pred_interval_sizes, 
        '.', 
        alpha = 0.05
    )
    ax1.set_title("pred set size vs y_true")
    ax1.set_xlabel("True label, $y_{true}$")
    ax1.set_ylabel(r"Prediction set size $|\tau(X)|$")


    ax2 = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    ax2.plot(
        np.abs(test_y-result.y_pred), 
        y_pred_interval_sizes, 
        '.', 
        alpha = 0.05
    )
    ax2.set_title("pred set size vs true absolute difference")
    ax2.set_xlabel("$|y_{pred}-y_{true}$|")
    ax2.set_ylabel(r"Prediction set size $|\tau(X)|$")


    ax3 = plt.subplot2grid((4, 5), (0, 2), rowspan=4, colspan=3)
    plt.scatter(
        test_X_pca[:,0],
        test_X_pca[:,1],
        edgecolors='none',
        s=6, 
        alpha = alpha,
        c=y_pred_interval_sizes
    )
    plt.colorbar()
    ax3.set_title('pred set sizes on first two PCs')
    ax3.set_xlabel("PC 1")
    ax3.set_ylabel("PC 2")
    ax3.plot()
    plt.suptitle(result.cp_model.name, fontsize = 15)
    plt.tight_layout()
    plt.show()
    
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
