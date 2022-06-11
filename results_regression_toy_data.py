#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from Models import MultipleQuantileRegressor
from CP import RegressionAdaptiveSquaredError, RegressionAdaptiveQuantile
from GP.gaussian_process_wrapper import GaussianProcessModelWrapper

from Toolbox.tools import multiple_split
from Toolbox.kernels import mahalanobis_sqe, squared_exponential

#%% create data sets 
# Wrapper for Linear regression so that it has a call function 
class LinearRegressionWrapper(LinearRegression):
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

class GPWrapper(GaussianProcessModelWrapper): # TODO fix Guassian process wrapper so it standardizes it self and transforms back in predict. 
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

# Hyper parameters 
alpha = 0.05
features_squared = lambda x : (x, x**2)
get_squared_features = lambda x : np.hstack(features_squared(x))
get_features = [lambda x: x, get_squared_features]
feature_names = ["no transform", "squared"]
kernels = [None, squared_exponential(0.1)]
kernel_names = ["Static", "Adaptive Squared Exponential"]
model_names = ["Linear regression", "Quantile Regression", "Guassian Process"]
runs = 100
size = 1000
train = 250
cali = 250
test = size - train - cali
train_cali_test = np.array([train, cali, test])/size

a = 3
b = -5
c = 3
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
        lm.fit(i(X), y)
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

    # # linear no feature transform
    # lm = LinearRegressionWrapper(n_jobs = 6)
    # lm.fit(X, y)

    # # squared features linear model 
    # lm_squared = LinearRegressionWrapper(n_jobs = 6)
    # lm_squared.fit(X_squared, y)

    # quantile regression no feature transforms 
    # qr = MultipleQuantileRegressor(X, y, quantiles = [alpha/2, 0.5, 1-alpha/2])

    # # quantile regression squared feature transforms 
    # qr_squared = MultipleQuantileRegressor(X_squared, y, quantiles = [alpha/2, 0.5, 1-alpha/2])

    # Gaussian Process
    # gp = GPWrapper(None, X, y, alpha)

    # # Guassian Process with squared features 
    # gp_squared = GPWrapper(None, X_squared, y, alpha)

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
                #scores.append(RegressionAdaptiveSquaredError(model, X_cali, y_cali, alpha, kernel=kernel, verbose=verbose))
                if type(model) == LinearRegressionWrapper:
                    scores.append(RegressionAdaptiveSquaredError(model, get_features[i](X_cali), y_cali, alpha, kernel=kernel, verbose=verbose))
                else:
                    scores.append(RegressionAdaptiveQuantile(model, get_features[i](X_cali), y_cali, alpha, kernel=kernel, verbose=verbose))
            cp_feature.append(scores) 
        cp_models.append(cp_feature)
    # cp_static_lm = RegressionAdaptiveSquaredError(lm, X_cali, y_cali, alpha,  verbose=True)
    # cp_static_lm_squared = RegressionAdaptiveSquaredError(lm_squared, X_cali_squard, y_cali, alpha,  verbose=True)

    # cp_anobis_lm = RegressionAdaptiveSquaredError(lm, X_cali, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
    # cp_anobis_lm_squared = RegressionAdaptiveSquaredError(lm_squared, X_cali_squard, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)

    # cp_static_qr = RegressionAdaptiveQuantile(qr, X_cali, y_cali, alpha,  verbose=True)
    # cp_static_qr_squared = RegressionAdaptiveQuantile(qr_squared, X_cali_squard, y_cali, alpha,  verbose=True)

    # cp_anobis_qr = RegressionAdaptiveQuantile(qr, X_cali, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
    # cp_anobis_qr_squared = RegressionAdaptiveQuantile(qr_squared, X_cali_squard, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
    
    # cp_static_gp = RegressionAdaptiveQuantile(gp, X_cali, y_cali, alpha,  verbose=True)
    # cp_static_gp_squared = RegressionAdaptiveQuantile(gp_squared, X_cali_squard, y_cali, alpha,  verbose=True)

    # cp_anobis_gp = RegressionAdaptiveQuantile(gp, X_cali, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)
    # cp_anobis_gp_squared = RegressionAdaptiveQuantile(gp_squared, X_cali_squard, y_cali, alpha,  kernel = squared_exponential(0.1), verbose=True)


    # cp_models = [cp_static_lm, cp_anobis_lm, cp_static_qr, cp_anobis_qr, cp_static_gp, cp_anobis_gp]
    # cp_models_squared = [cp_static_lm_squared, cp_anobis_lm_squared, cp_static_qr_squared, cp_anobis_qr_squared, cp_static_gp_squared, cp_anobis_gp_squared]
    return cp_models

# def plot_results(model, X_grid, caption=None, color=None):
#     preds = model(X_grid)
#     plt.plot(X_grid, preds[0], color=color)
#     plt.plot(X_grid, preds[1][:, 0], color=color)
#     plt.plot(X_grid, preds[1][:, 1], color=color)
#     plt.title(caption)

# def plot_results_squared(model, X_grid, caption=None, color=None):
#     preds = model(X_grid)
#     plt.plot(X_grid[:, 0], preds[0], color=color)
#     plt.plot(X_grid[:, 0], preds[1][:, 0], color=color)
#     plt.plot(X_grid[:, 0], preds[1][:, 1], color=color)
#     plt.title(caption + " Squared Features")

def plot_cp_model(model, X_grid, transform, caption=None, color=None):
    preds = model(transform(X_grid))
    plt.plot(X_grid, preds[0], color=color)
    plt.plot(X_grid, preds[1][:, 0], color=color)
    plt.plot(X_grid, preds[1][:, 1], color=color)
    plt.title(caption)

X_grid = np.arange(0, 1, 0.01).reshape(-1, 1)
#X_grid_squared = np.hstack((X_grid, X_grid**2))
colors = ["b", "k", "g", "r", "m", "y"] # TODO make this better 

#%% Show the what happens each iteration (show how the models fit - both adaptive and non)
datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "homosedatic"], 
            [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "hetrosedatic"], 
            [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "discontinuous"]]

for dataset in datasets: 
    # For all three data sets generate four plots. Two for non squared features two for squared features. 
    [y_train, y_cali, y_test, data_name] = dataset
    # Fit the models 
    models = load_models(X, y_train, alpha=alpha, features=get_features)

    # Fit CP models 
    cp_models = fit_cp(models, X_cali, y_cali, features=get_features, kernels=kernels, verbose=True)

    for feature_n, feature in enumerate(cp_models): 
        
        for model_n, models in enumerate(feature): 
            for kernel_n, model in enumerate(models):
                plt.subplot(len(feature), len(kernels), 1 + model_n*len(kernels) + kernel_n)
                plt.plot(X_test, dataset[2], ',')
                plot_cp_model(model, X_grid, transform=get_features[feature_n], caption=f"{model_names[model_n]} {kernel_names[kernel_n]}", color=None) 
        plt.suptitle(f"{data_name} {feature_names[feature_n]}")
        plt.tight_layout()
        plt.savefig(f"./results/{data_name} {feature_names[feature_n]}")
        plt.clf()
    
    for feature_n, feature in enumerate(cp_models): 
        for kernel_n in range(len(kernels)):
            for model_n, models in enumerate(feature): 
                plot_cp_model(models[kernel_n], X_grid, transform=get_features[feature_n], color=colors[model_n*len(kernels) + kernel_n]) 
            plt.plot(X_test, dataset[2], ',')
            plt.suptitle(f"All models {data_name} {feature_names[feature_n]} {kernel_names[kernel_n]}")
            plt.tight_layout()
            plt.savefig(f"./results/All models {data_name} {feature_names[feature_n]} {kernel_names[kernel_n]}")
            plt.clf()


# for dataset in datasets: 
#     # For all three data sets generate four plots. Two for non squared features two for squared features. 
#     models, cp_models, cp_models_squared = load_models(X, dataset[0], dataset[1], alpha)

#     for i in range(len(cp_models)):
#         plt.subplot(3,2, i+1)
#         plt.plot(X_test, dataset[2], ',')
#         plot_results(cp_models[i], X_grid, caption=cp_model_names[i])
    
#     plt.tight_layout()
#     plt.savefig(f"./results/{dataset[3]}_side_by_side")
#     plt.clf()
    
#     for i in range(len(cp_models)):
#         plot_results(cp_models[i], X_grid, caption=cp_model_names[i], color=colors[i])
#     plt.plot(X_test, dataset[2], ',')
    
#     plt.tight_layout()
#     plt.savefig(f"./results/{dataset[3]}_six_in_one")
#     plt.clf()

#     for i in range(len(cp_models_squared)):
#         plt.subplot(3,2, i+1)
#         plt.plot(X_test, dataset[2], ',')
#         plot_results_squared(cp_models_squared[i], X_grid_squared, caption=cp_model_names[i])
    
#     plt.tight_layout()
#     plt.savefig(f"./results/{dataset[3]}_side_by_side_squared")
#     plt.clf()
    
#     for i in range(len(cp_models_squared)):
#         plot_results_squared(cp_models_squared[i], X_grid_squared, caption=cp_model_names[i], color=colors[i])
#     plt.plot(X_test, dataset[2], ',')
    
#     plt.tight_layout()
#     plt.savefig(f"./results/{dataset[3]}_six_in_one_squared")
#     plt.clf()

    # TODO : make some plots that shows how the different distributions change 

#%% Run all models on the three data sets runs times 

coverage = np.zeros((len(cp_model_names)*2, runs, len(datasets)))

for i in range(runs):
    X, X_cali, X_test, y_homosedatic, y_homosedatic_cali, y_homosedatic_test, y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, y_discontinuous, y_discontinuous_cali, y_discontinuous_test = create_data_set(size, train, cali, a, b, c, noise)
    X_test_squared = np.hstack((X, X**2))

    datasets = [[y_homosedatic, y_homosedatic_cali, y_homosedatic_test, "homosedatic"], 
                [y_hetrosedatic, y_hetrosedatic_cali, y_hetrosedatic_test, "hetrosedatic"], 
                [y_discontinuous, y_discontinuous_cali, y_discontinuous_test, "discontinuous"]]

    for k, dataset in enumerate(datasets): 
        models, cp_models, cp_models_squared = load_models(X, dataset[0], dataset[1], alpha)

        for j, model in enumerate(cp_models): 
            y_preds, pred_intervals, in_pred_set, empirical_coverage, effective_sample_sizes = model.evaluate_coverage(X_test, dataset[2])

            coverage[j, i, k] = empirical_coverage
        

        for j, model in enumerate(cp_models_squared): 
            y_preds, pred_intervals, in_pred_set, empirical_coverage, effective_sample_sizes = model.evaluate_coverage(X_test_squared, dataset[2])

            coverage[j + len(cp_model_names), i, k] = empirical_coverage
