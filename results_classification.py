#%%
import numpy as np
import matplotlib.pyplot as plt

from CP import Classification_cumulative_softmax
from CP.CP_base import CPEvalData
#from GP.gaussian_process_wrapper import GaussianProcessModelWrapper
import torch
import torch.nn
from model import CNN_class, train, test

from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm

#from Toolbox.tools import multiple_split
#from Toolbox.kernels import mahalanobis_exponential, exponential
from scipy.stats import norm
import os 
#plt.rcParams['text.usetex'] = True


if not "results" in os.listdir("."):
    os.mkdir("./results")
if not "EMNIST" in os.listdir("./results"):
    os.mkdir("./results/EMNIST")
if not "ADULT" in os.listdir("./results"):
    os.mkdir("./results/ADULT ")

#%% create data sets 
def load_data(batch_size = 500, data_workers = 1, train_points = 1000, cali_points = 200, test_points = 2000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()]) # transforms.LinearTransformation(torch.Tensor([[1, 0, 0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]), torch.Tensor

    training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)

    # Take train and cali points from the training set 
    train_set, cali_set = train_test_split(np.arange(len(training_set)), train_size=train_points, test_size=cali_points, stratify=training_set.targets)
    train_set = data_utils.Subset(training_set, train_set)
    cali_set = data_utils.Subset(training_set, cali_set)
    #train_set, cali_set = train_test_split(train_cali_set, test_size=)

    # Test set taken from the test set. 
    test_indices, _ = train_test_split(np.arange(len(test_set)), train_size=test_points, stratify=test_set.targets)
    test_set = data_utils.Subset(test_set, test_indices)

    train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=data_workers)
    cali_dl = DataLoader(cali_set, batch_size=batch_size, num_workers=data_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=data_workers)

    return train_dl, cali_dl, test_dl


def plot_letters(n, dl):
    # Plot letters for report 
    batch = next(iter(dl))
    imgs = batch[0].detach().numpy()

    fig, axs = plt.subplots(1, n)
    fig.suptitle('Handwritten Letters')
    for i in range(n):
        plt.sca(axs[i])
        plt.imshow(imgs[i].reshape(28, 28).T, interpolation='nearest')

class CNN():
    # Loads CNN, trains it if it hasn't been and implements a call method 
    def __init__(self, width, depth, lr=None, wd=None, train_dl=None, params=None, alpha=0.9):
        self.model = CNN_class(width, depth)
        self.alpha = alpha

        if params != None: 
            self.model.load_state_dict(torch.load('bayesian_optimization_best_model.pt'))
        else: 
            train(self.model, train_dl, lr, wd)
        
    def __call__(self, X):
        predictions = torch.nn.functional.softmax(self.model(torch.permute(X, (0,1,3,2))), dim=1).detach().numpy()
        return predictions

class Coverage_Wrapper(): 
    # Makes it possible to evaluate models without applying CP 
    def __init__(self, model):
        self.model = model 
        self.alpha = self.model.alpha 

    def __call__(self, X):
        predictions = model(X)
        indices = np.argsort(-predictions, axis=1)
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(predictions, indices, axis=1)
        weighted_preds = np.cumsum(sorted, axis = 1)
        weighted_preds = np.take_along_axis(weighted_preds, reverse_indices, axis=1)
        
        pred = np.argmin(weighted_preds)
        preds = weighted_preds <= alpha
        return pred, preds
        
    def eval_coverage(self, X, y):
        pred, preds = self(X) 
        pred_sizes = np.sum(preds, axis=1)
        avg_size = np.mean(pred_sizes)

        accuracy = np.mean(pred == y) 
        empirical_covarage = np.mean([y[i] in preds[i] for i in range(len(y))])
        return pred_sizes, avg_size, accuracy, empirical_covarage


# Hyper parameters 
fig_size = (13, 8)
alpha = 0.10
#kernels = [None, exponential(0.01)]
#kernel_names = ["Normal CP", "Naive LCP Exponential"]

#colors = ["k", "r", "b", "g", "m", "y"] 

runs = 1000
size = 3200
train_points = 1000
cali_points = 200
test_points = size - train_points - cali_points
train_cali_test = np.array([train_points, cali_points, test_points])/size


#%% Show one instance of the data set 
train_dl, cali_dl, test_dl = load_data(train_points=train_points, cali_points=cali_points, test_points=test_points)
plt.figure(figsize=fig_size, dpi=200)
#plt.rcParams["figure.figsize"] = (10, 5)
plot_letters(7, train_dl)
plt.tight_layout()
plt.savefig("./results/EMNIST/handwritten_letters")
plt.show()

#%% load models 
#Getting correct hyper parameters
validation_accuracies = np.load('bayesian_optimization_accuracies_val.npy')
hyperparameters = np.load('bayesian_optimization_hyperparameters.npy')
hyperparameter = hyperparameters[np.argmax(validation_accuracies)]

train_dl, calibration_dl, test_dl = load_data(train_points=train_points, cali_points=cali_points, test_points=test_points)

model = CNN(*hyperparameter, train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
# If train from scratch v 
#model = CNN(*hyperparameter, train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
evaluate_model = Coverage_Wrapper(model) 

# TODO put this into a CO thing. 

accuracies = []
for (x_cal,y_cal) in tqdm(calibration_dl):
    CP_model = Classification_cumulative_softmax(model=model1,calibration_set_x=x_cal,
                                                            calibration_set_y=y_cal.detach().numpy(),
                                                            alpha=0.05)
    temp_acc =[]
    for (x_val, y_val) in validation_dl:
        conf_intervals = CP_model.predict(x_val)

        acc = np.mean([y_hat[y] for y, y_hat in zip(y_val.detach().numpy(), conf_intervals)])
        temp_acc.append(acc)
    accuracies.append(temp_acc)

print(accuracies)

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
            # preds = models[model_n][feature_n](get_features[feature_n](X_grid))
            # plt.plot(X_grid, preds, color=colors[0])
            # plt.fill_between(X_grid[:, 0], preds[:, 0], preds[:, 2], alpha=0.1)
            plot_model(models[model_n][feature_n], X_grid, get_features[feature_n], caption=f"{model_names[model_n]}", color="k", coverage=print_coverage, X_test=X_test, y_test=y_test)
            if show_test: 
                plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
            if show_cali: 
                plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
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
        plt.savefig(f"./results/stand_alone/{data_name}_{feature_names[feature_n]}")
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
            plt.savefig(f"./results/all_models/{data_name}_{feature_names[feature_n]}_{kernel_names[kernel_n]}")
            plt.clf()

    # Plot one model with its different CP configurations applied 
    for feature_n, feature in enumerate(cp_models):
        for model_n, model_type in enumerate(feature):
            plt.figure(figsize=fig_size, dpi=200)
            #plt.rcParams["figure.figsize"] = (fig_size)
            for kernel_n, cp_model in enumerate(model_type):
                plot_cp_model(cp_model, X_grid, transform=get_features[feature_n], color=colors[kernel_n + 1], label=kernel_names[kernel_n], coverage=print_coverage, X_test=X_test, y_test=y_test)
            # preds = models[model_n][feature_n](get_features[feature_n](X_grid))
            # plt.plot(X_grid, preds, color=colors[0], label="Without CP")
            # plt.fill_between(X_grid[:, 0], preds[:, 0], preds[:, 2], alpha=0.2)
            plot_model(models[model_n][feature_n], X_grid, get_features[feature_n], label="Without CP", color=colors[0], coverage=print_coverage, X_test=X_test, y_test=y_test)
            if show_test: 
                plt.plot(X_test, y_test, dot_style, alpha=test_a, color=test_col)
            if show_cali: 
                plt.plot(X_cali, y_cali, dot_style, alpha=cali_a, color=cali_col)
            plt.legend()
            plt.suptitle(f"Model {model_names[model_n]} with feature {feature_names[feature_n]} and dataset {data_name}")
            plt.tight_layout()
            plt.savefig(f"./results/one_model/{data_name}_{model_names[model_n]}_{feature_names[feature_n]}")
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

print(avg_coverage)
print(avg_pred_size)
# %%
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
            plt.savefig(f"./results/coverage_histograms/{title}")
            plt.clf()
