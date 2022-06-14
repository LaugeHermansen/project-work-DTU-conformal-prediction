#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Models import MultipleQuantileRegressor
from CP import RegressionAdaptiveSquaredError, RegressionAdaptiveQuantile
from GP.gaussian_process_wrapper import GaussianProcessModelWrapper

from Toolbox.tools import multiple_split
from Toolbox.kernels import mahalanobis_sqe, squared_exponential
from scipy.stats import norm

#%% create data sets 
# Wrapper for Linear regression so that it has a call function 
class LinearRegressionWrapper(LinearRegression):
    def fit(self, X, y, alpha): 
        super().fit(X, y)
        self.sd = np.std(super().predict(X) - y, ddof=1)
        self.alpha = alpha
        self.q = norm.ppf(1-alpha/2) 
    
    def predict(self, X): 
        preds = super().predict(X)
        return preds[:,None] + np.array([[-1, 0, 1]])*self.sd*self.q
        
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class GPWrapper(GaussianProcessModelWrapper): # TODO fix Guassian process wrapper so it standardizes it self and transforms back in predict. 
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

# Hyper parameters 
fig_size = (15, 5)
alpha = 0.05
get_normal_features = lambda x: x
get_squared_features = lambda x : np.hstack((x, x**2))
get_features = [get_normal_features, get_squared_features]
feature_names = ["No Feature Transform", "Squared Feature"]
kernels = [None, squared_exponential(0.1)]
kernel_names = ["Static", "Adaptive Squared Exponential"]
model_names = ["Linear Regression", "Quantile Regression", "Guassian Process"]

X_grid = np.arange(0, 1, 0.01).reshape(-1, 1) 
colors = ["k", "r", "b", "g", "m", "y"] 
dot_style = '.'

runs = 10
size = 3000
train = 1000
cali = 100
test = size - train - cali
train_cali_test = np.array([train, cali, test])/size

a = 3
b = -5
c = 5
noise = 3

def create_data_set(size, train, cali, a, b, c, noise):
    test = size - train - cali
    train_cali_test = np.array([train, cali, test])/size

    X = np.random.rand(size)
    x1 = 4*(X - 0.5)
    y_homosedatic = a*x1**2 + b*x1 + c + noise*np.random.randn(size)
    y_hetrosedatic = a*x1**2 + b*x1 + c + x1*noise*np.random.randn(size)
    y_discontinuous = x1*noise *np.random.randn(size) + a*x1**2 + c*(x1 < 0.5) + b*x1 + noise *np.random.randn(size)/4

    X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test, _,_,_ = multiple_split(train_cali_test, X, y_homosedatic, y_hetrosedatic, y_discontinuous, np.ones_like(y_hetrosedatic))
    X = X.reshape(-1,1)
    X_cali = X_cali.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    return X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test

#%% Show one instance of the data set 
X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test = create_data_set(size, train, cali, a, b, c, noise)
plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(1, 3, 1)
plt.plot(X, y_homosedatic, ',', color="b")
# plt.plot(X, y_homosedatic_cali, ',', color="g")
# plt.plot(X, y_homosedatic_test, ',', color="r")

plt.subplot(1, 3, 2)
plt.plot(X, y_hetrosedatic, ',', color="b")
# plt.plot(X, y_hetrosedatic_cali, ',', color="g")
# plt.plot(X, y_hetrosedatic_test, ',', color="r")


plt.subplot(1, 3, 3)
plt.plot(X, y_discontinuous, ',', color="b")
# plt.plot(X, y_discontinuous_cali, ',', color="g")
# plt.plot(X, y_discontinuous_test, ',', color="r")

plt.suptitle("The three toy data sets used for regression")
plt.tight_layout()
plt.savefig("./dataset")
plt.show()

#%% load models 
cp_model_names = ["Linear Regression", "Linear Regression Adaptive", "Quantile Regression", "Quantile Regression Adaptive", "Gaussian Process", "Gaussian Process Adaptive"]
def load_models(X, y, alpha, features=get_features):
    # get linear models 
    lms = []
    for i in get_features:
        lm = LinearRegressionWrapper(n_jobs = 6)
        lm.fit(i(X), y, alpha)
        lms.append(lm)

    # get quantile regression models 
    qrs = []
    for i in get_features:
        qr = MultipleQuantileRegressor(i(X), y, quantiles = [alpha/2, 0.5, 1-alpha/2])
        qrs.append(qr)

    # get Guassian Process models 
    gps = []
    for i in get_features:
        gp = GPWrapper(None, i(X), y, alpha)
        gps.append(gp)

    return [lms, qrs, gps]

def fit_cp(models, X_cali, y_cali, features=get_features, kernels=kernels, verbose=True):
    # Create fitted CP's with dimensions "features", "model", "CP score function" 
    cp_models = [] 
    for i, features in enumerate(features): 
        cp_feature = []
        for model in models:
            model = model[i]
            scores = []
            for kernel in kernels: 
                scores.append(RegressionAdaptiveQuantile(model, get_features[i](X_cali), y_cali, alpha, kernel=kernel, verbose=verbose))
            cp_feature.append(scores) 
        cp_models.append(cp_feature)
    return cp_models

def plot_cp_model(model, X_grid, transform, caption=None, color=None, label=None, grading=True):
    preds = model(transform(X_grid))
    plt.plot(X_grid, preds[0], color=color, label=label)
    plt.plot(X_grid, preds[1][:, 0], color=color)
    plt.plot(X_grid, preds[1][:, 1], color=color)
    if grading: 
        plt.fill_between(X_grid[:, 0], preds[1][:, 0], preds[1][:, 1], alpha=0.2, color=color)
    plt.title(caption)

#%% Show the what happens each iteration (show how the models fit - both adaptive and non)
datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "homoscedastic"], 
            [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "heteroscedastic"], 
            [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "discontinuous"]]

# 
for dataset in datasets: 
    # For all three data sets generate four plots. Two for non squared features two for squared features. 
    [y_train, y_cali, y_test, data_name] = dataset
    # Fit the models 
    models = load_models(X, y_train, alpha=alpha, features=get_features)

    # Fit CP models 
    cp_models = fit_cp(models, X_cali, y_cali, features=get_features, kernels=kernels, verbose=True)

    for feature_n, feature in enumerate(cp_models): 
        
        # Plot all models with their kernels and without CP 
        plt.rcParams["figure.figsize"] = (fig_size)
        for model_n, model_type in enumerate(feature): 
            # Plot the model without CP 
            plt.subplot(len(feature), len(kernels) + 1, 1 + model_n*(len(kernels) + 1))
            preds = models[model_n][feature_n](get_features[feature_n](X_grid))
            plt.plot(X_grid, preds, color=colors[0])
            plt.fill_between(X_grid[:, 0], preds[:, 0], preds[:, 2], alpha=0.2)
            plt.plot(X_test, y_test, dot_style)
            
            # Plot all different CP kernels applied
            for kernel_n, model in enumerate(model_type):
                plt.subplot(len(feature), len(kernels) + 1, 2 + model_n*(len(kernels) + 1) + kernel_n)
                plt.plot(X_test, y_test, dot_style)
                plot_cp_model(model, X_grid, transform=get_features[feature_n], caption=f"{model_names[model_n]} {kernel_names[kernel_n]}", color=None) 
        
        # Save image to results/stand_alone
        plt.suptitle(f"{data_name} {feature_names[feature_n]}")
        plt.tight_layout()
        plt.savefig(f"./results/stand_alone/{data_name}_{feature_names[feature_n]}")
        plt.clf()
    
    # Plot the models with the same CP applied 
    for feature_n, feature in enumerate(cp_models): 
        for kernel_n in range(len(kernels)):
            plt.rcParams["figure.figsize"] = (fig_size)
            for model_n, model_type in enumerate(feature): 
                plot_cp_model(model_type[kernel_n], X_grid, transform=get_features[feature_n], color=colors[model_n], label=model_names[model_n]) 
            
            # Save to resluts/all_models
            plt.plot(X_test, y_test, dot_style)
            plt.legend()
            plt.suptitle(f"All Models {data_name} {feature_names[feature_n]} {kernel_names[kernel_n]}")
            plt.tight_layout()
            plt.savefig(f"./results/all_models/{data_name}_{feature_names[feature_n]}_{kernel_names[kernel_n]}")
            plt.clf()

    # Plot one model with its different CP configurations applied 
    for feature_n, feature in enumerate(cp_models):
        for model_n, model_type in enumerate(feature):
            plt.rcParams["figure.figsize"] = (fig_size)
            for kernel_n, cp_model in enumerate(model_type):
                plot_cp_model(cp_model, X_grid, transform=get_features[feature_n], color=colors[kernel_n + 1], label=kernel_names[kernel_n])
            preds = models[model_n][feature_n](get_features[feature_n](X_grid))
            plt.plot(X_grid, preds, color=colors[0])
            plt.fill_between(X_grid[:, 0], preds[:, 0], preds[:, 2], alpha=0.2)
            plt.plot(X_test, y_test, dot_style)
            plt.legend()
            plt.suptitle(f"Model {model_names[model_n]} with feature {feature_names[feature_n]} and dataset {data_name}")
            plt.tight_layout()
            plt.savefig(f"./results/one_model/{data_name}_{model_names[model_n]}_{feature_names[feature_n]}")
            plt.clf()

#%% Run all models on the three data sets runs times 

coverage = np.zeros((3, len(feature_names), len(model_names), len(kernel_names), runs))
pred_size = np.zeros((3, len(feature_names), len(model_names), len(kernel_names), runs))

for run in range(runs):
    X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test = create_data_set(size, train, cali, a, b, c, noise)

    datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "homoscedastic"], 
                [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "heteroscedastic"], 
                [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "discontinuous"]]

    for dataset_n, dataset in enumerate(datasets): 
        # For all three data sets generate four plots. Two for non squared features two for squared features. 
        [y_train, y_cali, y_test, data_name] = dataset
        
        # Fit the models 
        models = load_models(X, y_train, alpha=alpha, features=get_features)

        # Fit CP models 
        cp_models = fit_cp(models, X_cali, y_cali, features=get_features, kernels=kernels, verbose=True)
        for feature_n, feature in enumerate(cp_models):
            for model_n, model_type in enumerate(feature):
                for kernel_n, cp_model in enumerate(model_type):
                    y_preds, pred_intervals, in_pred_set, empirical_coverage, effective_sample_sizes = cp_model.evaluate_coverage(get_features[feature_n](X_test), y_test)
                    avg_size = np.mean(pred_intervals[:, 1] - pred_intervals[:, 0]) 
                    coverage[dataset_n, feature_n, model_n, kernel_n, run] = empirical_coverage
                    pred_size[dataset_n, feature_n, model_n, kernel_n, run] = avg_size
      
avg_coverage = np.mean(coverage, axis=4)
avg_pred_size = np.mean(pred_size, axis=4)

print(avg_coverage)
print(avg_pred_size)
# %%
