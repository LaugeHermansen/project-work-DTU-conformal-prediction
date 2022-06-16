#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Models import MultipleQuantileRegressor
from CP import RegressionSquaredError, RegressionQuantile
from CP.CP_base import CPEvalData
#from GP.gaussian_process_wrapper import GaussianProcessModelWrapper
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from Toolbox.tools import multiple_split
from Toolbox.kernels import mahalanobis_exponential, exponential
from scipy.stats import t, norm
import os 
#plt.rcParams['text.usetex'] = True

np.random.seed(42)

if not "results" in os.listdir("."):
    os.mkdir("./results")
if not "final_toy_data" in os.listdir("./results"):
    os.mkdir("./results/final_toy_data")
if not "all_models" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/all_models")
if not "one_model" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/one_model")
if not "stand_alone" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/stand_alone")
if not "coverage_histograms" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/coverage_histograms")
if not "pred_histograms" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/pred_histograms")
if not "histograms" in os.listdir("./results/final_toy_data"):
    os.mkdir("./results/final_toy_data/histograms")
    
#%% create data sets 
# Wrapper for Linear regression so that it has a call function 
class LinearRegressionWrapper(LinearRegression):
    def fit(self, X, y, alpha): 
        super().fit(X, y)
        self.sd = np.std(super().predict(X) - y, ddof=1)
        self.n = len(y)
        self.Xinv = np.linalg.pinv(X.T @ X)
        self.alpha = alpha
        self.q = t.ppf(1-alpha/2, df=self.n-1) 
    
    def predict(self, X): 
        preds = super().predict(X)
        return preds[:,None] + self.q*np.array([[-1, 0, 1]])*self.sd*np.sqrt(1 + np.sum(X @ self.Xinv * X, axis=1))[:,None]
        #return preds[:,None] + np.array([[-1, 0, 1]])*self.sd*self.q
        
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class GPWrapper(GaussianProcessRegressor): 
    def __init__(self, X_train, y_train, alpha=1e-10, kernel=Matern() + WhiteKernel(), *, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, normalize_y=True, copy_X_train=True, random_state=None):
        super().__init__(kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer, normalize_y=normalize_y, copy_X_train=copy_X_train, random_state=random_state)
        super().fit(X_train, y_train)
    
    def predict(self, X): 
        y_mean, y_std = super().predict(X, return_std=True)
        pred_interval = y_mean[:, None] + np.array([-1, 0, 1]) * y_std[:,None] * norm.ppf(q = 1-self.alpha)
        return pred_interval
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class CoverageWrapper():
    def __init__(self, model):
        self.model = model
    
    def predict(self, X): 
        return self.model(X)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def evaluate_coverage(self, X, y):
        """
        Evaluate epirical coverage on test data points.
        
        Args:
        ------
            X: the features of the test data points
            
            y: the true labels/values of the test data points
        
        Returns:
        --------
            results: dataclass object containing:

                cp_model_name: str

                mean_effective_sample_size: int

                empirical_coverage: int

                y_preds: the predictions of the underlying model

                pred_sets: np.ndarray

                effective_sample_sizes: np.ndarray, float, shape = (n_test, 1)

                pred_set_sizes: np.ndarray, float, shape = (n_test, 1)

                kernel_outlier: np.ndarray, bool, shape = (n_test, 1)

                in_pred_set: np.ndarray, bool, shape = (n_test, 1)

        """
        
        # Compute results - These will be included in output 'result' if they don't begin with "_"
        
        # predictions from ml model, prediction sets, and effective sample sizes
        y_preds = self.predict(X)
        y_lower = y_preds[:, 0] 
        y_upper = y_preds[:, 2]
        y_preds = y_preds[:, 1]  

        # prediction set sizes
        pred_set_sizes = y_upper - y_lower

        # # boolean array - True if data point is an outlier according to kernel
        # # meaning that kernel(x) = 0 for all calibration points
        # kernel_outlier = effective_sample_sizes == np.array([None]*len(X))

        # mean_effective_sample_size = np.mean(effective_sample_sizes[~kernel_outlier].astype(float))
        in_pred_set = (y_lower <= y) * (y_upper >= y)
        empirical_coverage = np.mean(in_pred_set)

        result = CPEvalData(None, None, empirical_coverage, y_preds, np.vstack((y_lower, y_upper)), None, pred_set_sizes, None, in_pred_set)

        return result
        

# Hyper parameters 
fig_size = (8, 13)
alpha = 0.10
get_normal_features = lambda x: x
get_squared_features = lambda x : np.hstack((x, x**2))
get_features = [get_normal_features, get_squared_features]
feature_names = ["No Feature Transform", "Squared Feature"]
kernels = [None]
kernel_names = ["Normal CP"]
model_names = ["Linear Regression", "Quantile Regression", "Gaussian Process"]

X_grid = np.arange(0, 1, 0.01).reshape(-1, 1) 
colors = ["k", "r", "b", "g", "m", "y"] 
dot_style = '.'
show_cali = True 
show_test = True 
train_a = 0.9
cali_a = 0.8
test_a = 0.2
train_col = "b"
cali_col = "#9FFE36"#"g"
test_col = "r"
print_coverage = True

runs = 1000
size = 3000
train = 200
cali = 100
test = size - train - cali
train_cali_test = np.array([train, cali, test])/size

a = 3
b = -5
c = 15
noise = 6

def create_data_set(size, train, cali, a, b, c, noise):
    test = size - train - cali
    train_cali_test = np.array([train, cali, test])/size

    X = np.random.rand(size)
    x1 = 4*(X - 0.5)
    y_homosedatic = a*x1**2 + b*x1 + c + noise*np.random.rand(size)
    y_hetrosedatic = a*x1**2 + b*x1 + c + x1*noise*np.random.rand(size)
    y_discontinuous = x1*noise *np.random.rand(size) + a*x1**2 + c*(x1 < 0.5) + b*x1 + noise *np.random.rand(size)/4

    X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test, _,_,_ = multiple_split(train_cali_test, X, y_homosedatic, y_hetrosedatic, y_discontinuous, np.ones_like(y_hetrosedatic))
    X = X.reshape(-1,1)
    X_cali = X_cali.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    return X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test

#%% Show one instance of the data set 
X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test = create_data_set(size, train, cali, a, b, c, noise)
plt.figure(figsize=(13, 8), dpi=200)
#plt.rcParams["figure.figsize"] = (10, 5)

plt.subplot(1, 3, 1)
if show_test:
    plt.plot(X_test, y_homosedatic_test, dot_style, color=test_col, alpha=test_a)
plt.plot(X, y_homosedatic, dot_style, color=train_col, alpha=train_a)
if show_cali:
    plt.plot(X_cali, y_homosedatic_cali, dot_style, color=cali_col, alpha=cali_a)

plt.subplot(1, 3, 2)
if show_test:
    plt.plot(X_test, y_hetrosedatic_test, dot_style, color=test_col, alpha=test_a)
plt.plot(X, y_hetrosedatic, dot_style, color=train_col, alpha=train_a)
if show_cali:
    plt.plot(X_cali, y_hetrosedatic_cali, dot_style, color=cali_col, alpha=cali_a)

plt.subplot(1, 3, 3)
if show_test:
    plt.plot(X_test, y_discontinuous_test, dot_style, color=test_col, alpha=test_a)
plt.plot(X, y_discontinuous, dot_style, color=train_col, alpha=train_a)
if show_cali:
    plt.plot(X_cali, y_discontinuous_cali, dot_style, color=cali_col, alpha=cali_a)


plt.suptitle("The three toy data sets used for regression")
plt.tight_layout()
plt.savefig("./results/final_toy_data/datasets")
plt.show()

#%% load models 
#cp_model_names = ["Linear Regression", "Linear Regression Adaptive", "Quantile Regression", "Quantile Regression Adaptive", "Gaussian Process", "Gaussian Process Adaptive"]
def load_models(X, y, alpha, features=get_features):
    # get linear models 
    lms = []
    for i in get_features:
        lm = LinearRegressionWrapper(n_jobs = 6)
        lm.fit(i(X), y, alpha)
        lms.append(CoverageWrapper(lm))

    # get quantile regression models 
    qrs = []
    for i in get_features:
        qr = MultipleQuantileRegressor(i(X), y, quantiles = [alpha/2, 0.5, 1-alpha/2])
        qrs.append(CoverageWrapper(qr))

    # get Guassian Process models 
    gps = []
    for i in get_features:
        gp = GPWrapper(i(X), y, alpha)
        gps.append(CoverageWrapper(gp))

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
                scores.append(RegressionQuantile(model, get_features[i](X_cali), y_cali, alpha, kernel=kernel, verbose=verbose))
            cp_feature.append(scores) 
        cp_models.append(cp_feature)
    return cp_models

def plot_cp_model(model, X_grid, transform, caption=None, color=None, label=None, grading=True, coverage=None, X_test=None, y_test=None):
    preds = model(transform(X_grid))
    # Put the coverage in the upper corner or in the label 
    if coverage:
        results = model.evaluate_coverage(transform(X_test), y_test) 
        empirical_coverage = round(results.empirical_coverage, 2)
        if label != None:
            label = f"{label} {empirical_coverage}"
        else:
            plt.text(x=0.95,y=0.90*np.max(y_test), s=f"Empirical Coverage: {empirical_coverage}", ha="right", va="top")
    plt.plot(X_grid, preds[0], color=color, label=label)
    plt.plot(X_grid, preds[1][:, 0], color=color)
    plt.plot(X_grid, preds[1][:, 1], color=color)
    if grading: 
        plt.fill_between(X_grid[:, 0], preds[1][:, 0], preds[1][:, 1], alpha=0.1, color=color)
    plt.title(caption)
    
def plot_model(model, X_grid, transform, caption=None, color=None, label=None, grading=True, coverage=None, X_test=None, y_test=None):
    preds = model(transform(X_grid))
    # Put the coverage in the upper corner or in the label 
    if coverage:
        results = model.evaluate_coverage(transform(X_test), y_test) 
        empirical_coverage = round(results.empirical_coverage, 2)
        if label != None:
            label = f"{label} {empirical_coverage}"
        else:
            plt.text(x=0.95,y=0.90*np.max(y_test), s=f"Empirical Coverage: {empirical_coverage}", ha="right", va="top")
    plt.plot(X_grid, preds[:, 1], color=color, label=label)
    plt.plot(X_grid, preds[:, 0], color=color)
    plt.plot(X_grid, preds[:, 2], color=color)
    if grading: 
        plt.fill_between(X_grid[:, 0], preds[:, 0], preds[:, 2], alpha=0.1, color=color)
    plt.title(caption)

#%% Show the what happens each iteration (show how the models fit - both adaptive and non)
datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "Homoscedastic"], 
            [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "Heteroscedastic"], 
            [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "Discontinuous"]]

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
        plt.figure(figsize=(16, 9), dpi=200)
        for model_n, model_type in enumerate(feature): 
            # Plot the model without CP 
            plt.subplot(len(feature), len(kernels) + 1, 1 + model_n*(len(kernels) + 1))
            plot_model(models[model_n][feature_n], X_grid, get_features[feature_n], caption=f"{model_names[model_n]}", color="k", coverage=print_coverage, X_test=X_test, y_test=y_test)
            if show_test: 
                plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
            plt.plot(X, y_train, dot_style, alpha=train_a, color=train_col)
            # if show_cali: 
            #     plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
            #plt.title(f"{model_names[model_n]}")
            
            # Plot all different CP kernels applied
            for kernel_n, model in enumerate(model_type):
                plt.subplot(len(feature), len(kernels) + 1, 2 + model_n*(len(kernels) + 1) + kernel_n)
                if show_test: 
                    plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
                if show_cali: 
                    plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
                plot_cp_model(model, X_grid, transform=get_features[feature_n], caption=f"{model_names[model_n]} {kernel_names[kernel_n]}", color="k", coverage=print_coverage, X_test=X_test, y_test=y_test) 
            
        # Save image to results/stand_alone
        plt.suptitle(f"{data_name} {feature_names[feature_n]}")
        plt.tight_layout()
        plt.savefig(f"./results/final_toy_data/stand_alone/{data_name}_{feature_names[feature_n]}")
        plt.clf()
    
    # Plot the models with the same CP applied 
    for feature_n, feature in enumerate(cp_models): 
        for kernel_n in range(len(kernels)):
            plt.figure(figsize=fig_size, dpi=200)
            #plt.rcParams["figure.figsize"] = (fig_size)
            for model_n, model_type in enumerate(feature): 
                plot_cp_model(model_type[kernel_n], X_grid, transform=get_features[feature_n], color=colors[model_n], label=model_names[model_n], coverage=print_coverage, X_test=X_test, y_test=y_test) 
            
            # Save to resluts/all_models
            if show_test: 
                plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
            if show_cali: 
                plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
            plt.legend()
            plt.suptitle(f"All Models {data_name} {feature_names[feature_n]} {kernel_names[kernel_n]}")
            plt.tight_layout()
            plt.savefig(f"./results/final_toy_data/all_models/{data_name}_{feature_names[feature_n]}_{kernel_names[kernel_n]}")
            plt.clf()

    # Plot one model with its different CP configurations applied 
    for feature_n, feature in enumerate(cp_models):
        for model_n, model_type in enumerate(feature):
            plt.figure(figsize=fig_size, dpi=200)
            #plt.rcParams["figure.figsize"] = (fig_size)
            for kernel_n, cp_model in enumerate(model_type):
                plot_cp_model(cp_model, X_grid, transform=get_features[feature_n], color=colors[kernel_n + 1], label=kernel_names[kernel_n], coverage=print_coverage, X_test=X_test, y_test=y_test)
            plot_model(models[model_n][feature_n], X_grid, get_features[feature_n], label="Without CP", color=colors[0], coverage=print_coverage, X_test=X_test, y_test=y_test)
            if show_test: 
                plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
            if show_cali: 
                plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
            plt.legend()
            plt.suptitle(f"Model {model_names[model_n]} with feature {feature_names[feature_n]} and dataset {data_name}")
            plt.tight_layout()
            plt.savefig(f"./results/final_toy_data/one_model/{data_name}_{model_names[model_n]}_{feature_names[feature_n]}")
            plt.clf()

#%% Run all models on the three data sets runs times 

coverage = np.zeros((3, len(feature_names), len(model_names), len(kernel_names) + 1, runs))
pred_size = np.zeros((3, len(feature_names), len(model_names), len(kernel_names) + 1, runs))

for run in range(runs):
    X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test = create_data_set(size, train, cali, a, b, c, noise)

    datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "Homoscedastic"], 
                [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "Heteroscedastic"], 
                [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "Discontinuous"]]

    for dataset_n, dataset in enumerate(datasets): 
        # For all three data sets generate four plots. Two for non squared features two for squared features. 
        [y_train, y_cali, y_test, data_name] = dataset
        
        # Fit the models 
        models = load_models(X, y_train, alpha=alpha, features=get_features)

        # Fit CP models 
        cp_models = fit_cp(models, X_cali, y_cali, features=get_features, kernels=kernels, verbose=True)
        for feature_n, feature in enumerate(cp_models):
            for model_n, model_type in enumerate(feature):
                # Model without CP
                result = models[model_n][feature_n].evaluate_coverage(get_features[feature_n](X_test), y_test)
                avg_size = np.mean(result.pred_set_sizes) 
                coverage[dataset_n, feature_n, model_n, 0, run] = result.empirical_coverage
                pred_size[dataset_n, feature_n, model_n, 0, run] = avg_size
                
                for kernel_n, cp_model in enumerate(model_type):
                    # Model with applied CP framework 
                    result = cp_model.evaluate_coverage(get_features[feature_n](X_test), y_test)
                    avg_size = np.mean(result.pred_set_sizes) 
                    coverage[dataset_n, feature_n, model_n, kernel_n + 1, run] = result.empirical_coverage
                    pred_size[dataset_n, feature_n, model_n, kernel_n + 1, run] = avg_size
      
avg_coverage = np.mean(coverage, axis=4)
avg_pred_size = np.mean(pred_size, axis=4)

coverage_path = "./results/final_toy_data/histograms/coverage.npy"
pred_size_path = "./results/final_toy_data/histograms/pred_size.npy"
np.save(coverage_path, coverage)
np.save(pred_size_path, pred_size)

#print(avg_coverage)
#print(avg_pred_size)
# %%
# coverage = np.load(coverage_path)
# pred_size = np.load(pred_size_path)

def plot_histogram(data, colors, legends, title, trans=0.5, bins=20):
    for i, d in enumerate(data): 
        print(d, bins, trans, legends, colors)
        plt.hist(d, bins, alpha=trans, label=legends[i], color=colors[i])
    plt.legend()
    plt.title(title)

for dataset_n, dataset in enumerate(coverage): 
    for feature_n, feature in enumerate(dataset): 
        for model_n, model in enumerate(feature): 
            title = f"Coverage For {model_names[model_n]} With {feature_names[feature_n]} On {datasets[dataset_n][3]}"
            legends = ["Without CP"]
            legends.extend(kernel_names)
            plot_histogram(model, colors=colors, legends=legends, title=title)
            plt.savefig(f"./results/final_toy_data/coverage_histograms/{title}")
            plt.clf()
