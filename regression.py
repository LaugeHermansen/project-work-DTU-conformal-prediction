#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from CP import RegressionAdaptiveSquaredError
from plot_helpers import barplot
#%%

#Read data

data = pd.read_csv('data/CASP.csv', header = 0)

# test for missing data
for c in data.columns:
    print(f"{c}: nans: {np.sum(np.isnan(data[c]))}, {list(set(data[c].apply(type)))}, [{min(data[c])}, {max(data[c])}]")

#split x and y, and standardize x

X = data[data.columns[1:]].to_numpy()
y = data[data.columns[0]].to_numpy().squeeze()

st = StandardScaler()
st.fit(X)
X_standard = st.transform(X)

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

train_X, temp_X, train_y, temp_y, train_stratify, temp_stratify = train_test_split(X_standard, y, stratify, test_size=0.5, stratify=stratify, shuffle = True)
cal_X, test_X, cal_y, test_y, cal_stratify, test_stratify = train_test_split(temp_X, temp_y,train_stratify, test_size=0.5, stratify=temp_stratify, shuffle = True)

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

# fit simple model (linear regression) and run CP
lm = LinearRegression(n_jobs = 6)
lm.fit(train_X, train_y)

def squared_exponential(X1, X2):
    for x in X2:
        yield np.exp(-np.sum((X1-x)**2/0.5, axis = 1))

# I changed the adaptive regression base a bit
# Now, if you don't specify a kernel, it is just
# the standard CP, but if you specify a kernel,
# it is adaptive

cplm_ad = RegressionAdaptiveSquaredError(lm, cal_X, cal_y, 0.2, 'predict', kernel = squared_exponential, verbose = True)
cplm_st = RegressionAdaptiveSquaredError(lm, cal_X, cal_y, 0.2, 'predict')

y_preds_ad, pred_intervals_ad, in_pred_set_ad, empirical_coverage_ad = cplm_ad.evaluate_coverage(test_X, test_y)
y_preds_st, pred_intervals_st, in_pred_set_st, empirical_coverage_st = cplm_st.evaluate_coverage(test_X, test_y)


#%%
# plot a bunch of stuff

print(empirical_coverage_ad)
print(empirical_coverage_st)


plt.plot(test_y, pred_intervals_ad[:,1] - pred_intervals_ad[:,0], '.', alpha = 0.05)
plt.title("pred set size vs y_true")
plt.xlabel("True label, $y_{true}$")
plt.ylabel(r"Prediction set size $|\tau(X)|$")
plt.show()


plt.plot(np.abs(test_y-y_preds_ad), pred_intervals_ad[:,1] - pred_intervals_ad[:,0], '.', alpha = 0.05)
plt.title("pred set size vs true absolute difference")
plt.xlabel("$|y_{pred}-y_{true}$|")
plt.ylabel(r"Prediction set size $|\tau(X)|$")
plt.show()


test_X_pca = pca.transform(test_X)
plt.scatter(test_X_pca[:,0],test_X_pca[:,1],edgecolors='none',s=6, alpha = 0.2,c=pred_intervals_ad[:,1] - pred_intervals_ad[:,0])
plt.colorbar()
plt.title('pred set sizes on first two PCs')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.plot()
plt.show()

#%%

# Plot the coverage based on class (y-intervals)

bar = []
height_ad = []
height_st = []
for y_class in sorted(set(stratify)):
    bar.append(y_class)
    height_ad.append(np.mean(in_pred_set_ad[test_stratify == y_class]))
    height_st.append(np.mean(in_pred_set_st[test_stratify == y_class]))


# barplot(bar, (height_ad, height_st), ("adaptive", "static"))

plt.plot(bar, height_st, label = "static")
plt.plot(bar, height_ad, label = "adaptive")
print(np.std(height_ad))
print(np.std(height_st))
plt.legend()
plt.title("Coverage vs y-intervals")
plt.show()
