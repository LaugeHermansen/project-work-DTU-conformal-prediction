#%%
import numpy as np
import matplotlib.pyplot as plt
from sympy import hyper

from CP import ClassificationCumulativeSoftmax, ClassificationSoftmax
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
from scipy.stats import norm, betabinom
import os 
import pickle
from Toolbox.plot_helpers import *
#plt.rcParams['text.usetex'] = True


if not "results" in os.listdir("."):
    os.mkdir("./results")
if not "EMNIST" in os.listdir("./results"):
    os.mkdir("./results/EMNIST")

#%% create data sets 
def load_data(batch_size = 500, data_workers = 1, train_points = 1000, cali_points = 200, test_points = 2000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()]) # transforms.LinearTransformation(torch.Tensor([[1, 0, 0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]), torch.Tensor

    training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)

    # Take train and cali points from the training set 
    #train_set, _ = train_test_split(np.arange(len(training_set)), train_size=train_points, stratify=training_set.targets)
    #train_set = data_utils.Subset(training_set, train_set)
    #cali_set = data_utils.Subset(training_set, cali_set)
    #train_set, cali_set = train_test_split(train_cali_set, test_size=)

    # Test set taken from the test set. 
    test_indices, cali_indices = train_test_split(np.arange(len(test_set)), train_size=0.5, stratify=test_set.targets)
    cali_set = data_utils.Subset(test_set, cali_indices)
    test_set = data_utils.Subset(test_set, test_indices)
    

    train_dl = DataLoader(training_set, batch_size=train_points)#, num_workers=data_workers)
    cali_dl = DataLoader(cali_set, batch_size=cali_points)#, num_workers=data_workers)
    test_dl = DataLoader(test_set, batch_size=test_points)#, num_workers=data_workers)

    return train_dl, cali_dl, test_dl

def plot_letters(n, dl):
    # Plot letters for report 
    batch = next(iter(dl))
    imgs = batch[0].detach().numpy()

    #plt.figure(figsize=(7*n, 7), dpi=200)

    fig, axs = plt.subplots(1, n)
    fig.suptitle('Handwritten Letters')
    for i in range(n):
        plt.sca(axs[i])
        plt.imshow(imgs[i].reshape(28, 28).T, interpolation='nearest')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

class CNN():
    # Loads CNN, trains it if it hasn't been and implements a call method 
    def __init__(self, width, depth, lr=None, wd=None, train_dl=None, params=None, alpha=0.9):
        self.model = CNN_class(width, depth)
        self.alpha = alpha

        if params != None: 
            self.model.load_state_dict(params)
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
        predictions = self.model(X)
        indices = np.argsort(-predictions, axis=1)
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(predictions, indices, axis=1)
        weighted_preds = np.cumsum(sorted, axis = 1)
        weighted_preds = np.take_along_axis(weighted_preds, reverse_indices, axis=1)
        
        pred = np.argmin(weighted_preds, axis=1)
        preds = weighted_preds < 1-alpha
        weighted_preds[preds] += 1
        preds[np.arange(len(preds)), np.argmin(weighted_preds, axis=1)] = 1
        
        return pred, preds
        
    def eval_coverage(self, X, y):
        pred, preds = self(X) 
        pred_sizes = np.sum(preds, axis=1)
        avg_size = np.mean(pred_sizes)

        accuracy = np.mean(pred == y) 
        empirical_covarage = np.mean([preds[i][y[i]] for i in range(len(y))])
        return pred_sizes, avg_size, accuracy, empirical_covarage


# Hyper parameters 
fig_size = (13, 8)
alpha = 0.05
#kernels = [None, exponential(0.01)]
#kernel_names = ["Normal CP", "Naive LCP Exponential"]

#colors = ["k", "r", "b", "g", "m", "y"] 

size = 2000
train_points = 1000
cali_points = 500
test_points = size - train_points - cali_points
train_cali_test = np.array([train_points, cali_points, test_points])/size


#%% Show one instance of the data set 
train_dl, cali_dl, test_dl = load_data(train_points=train_points, cali_points=cali_points, test_points=test_points)
#plt.rcParams["figure.figsize"] = (10, 5) # TODO fix so that there is not all that white space
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

model = CNN(width=hyperparameter[2], depth=hyperparameter[3], lr=hyperparameter[0], wd=hyperparameter[1], train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
# If train from scratch v 
#model = CNN(*hyperparameter, train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
evaluate_model = Coverage_Wrapper(model) 

#cali_dl = iter(calibration_dl)

results_EMNIST = {"model pred sizes" : [],
                  "model avg size" : [], 
                  "model accuracy" : [],
                  "model empirical coverage" : [], 
                  "CP softmax pred sizes" : [],
                  "CP cumulative pred sizes" : [], 
                  "CP softmax avg size" : [], 
                  "CP cumulative avg size" : [], 
                  "CP softmax empirical coverage" : [],
                  "CP cumulative empirical coverage" : [] 
}

for (x_test, y_test), (calibration_set_x, calibration_set_y), (x_train, y_train) in tqdm(zip(test_dl, calibration_dl, train_dl)):

    y_test = y_test.detach().numpy()
    calibration_set_y = calibration_set_y.detach().numpy()
    # Apply CP framework 
    #calibration_set_x, calibration_set_y = next(iter(calibration_dl))
    CP_softmax = ClassificationSoftmax(model, calibration_set_x, calibration_set_y, alpha)
    CP_cumulative = ClassificationCumulativeSoftmax(model, calibration_set_x, calibration_set_y, alpha)
    
    # evaluate model and CP models 
    model_pred_sizes, model_avg_size, model_accuracy, model_empirical_covarage = evaluate_model.eval_coverage(x_test, y_test) 
    softmax_results = CP_softmax.evaluate_coverage(x_test, y_test) 
    cumulative_results = CP_cumulative.evaluate_coverage(x_test, y_test)

    # Save results 
    results_EMNIST["model pred sizes"].append(model_pred_sizes)
    results_EMNIST["model avg size"].append(model_avg_size)
    results_EMNIST["model accuracy"].append(model_accuracy)
    results_EMNIST["model empirical coverage"].append(model_empirical_covarage)
    results_EMNIST["CP softmax pred sizes"].append(softmax_results.pred_set_sizes)
    results_EMNIST["CP cumulative pred sizes"].append(cumulative_results.pred_set_sizes)
    results_EMNIST["CP softmax avg size"].append(np.mean(softmax_results.pred_set_sizes))
    results_EMNIST["CP cumulative avg size"].append(np.mean(cumulative_results.pred_set_sizes))
    results_EMNIST["CP softmax empirical coverage"].append(softmax_results.empirical_coverage)
    results_EMNIST["CP cumulative empirical coverage"].append(cumulative_results.empirical_coverage)


results_EMNIST_file = open("./results/EMNIST/results_EMNIST.pkl", "wb")
pickle.dump(results_EMNIST, results_EMNIST_file)
results_EMNIST_file.close()
#%%
# Show the results 

results_EMNIST_file = open("./results/EMNIST/results_EMNIST.pkl", "rb")
results_EMNIST = pickle.load(results_EMNIST_file)

#V2 
plt.figure(figsize=(26, 18), dpi=200)
plt.rc("font", size=28)
# Model pred sizes 
plt.plot(*compute_barplot_data(np.hstack(results_EMNIST["model pred sizes"])), "-o", alpha=0.5, linewidth=7, markersize=12, color="r", label="Cumulative model probability")
# CP softmax prediction sizes 
plt.plot(*compute_barplot_data(np.hstack(results_EMNIST["CP softmax pred sizes"])), "-o", alpha=0.5, linewidth=7, markersize=12, color="b", label="Normal CP")
# CP cumulative sum prediction sizes 
plt.plot(*compute_barplot_data(np.hstack(results_EMNIST["CP cumulative pred sizes"])), "-o", alpha=0.5, linewidth=7, markersize=12, color="y", label="Adaptive CP")

plt.title("Prediction Set Sizes") 
plt.xlabel("Prediction set sizes")
plt.ylabel("Volume")
plt.legend()
plt.savefig("./results/EMNIST/pred_set_sizesYoink")
plt.clf()

# Plot the three histograms together 
def plot_histogram(data, colors, legends, title, trans=0.5, bins=14):
    for i, d in enumerate(data): 
        plt.hist(d, bins, density=True, alpha=trans, label=legends[i], color=colors[i])
    plt.legend()
    plt.title(title)
    plt.xlabel("Empirical Coverage")
    plt.ylabel("Density")

plt.figure(figsize=(26, 18), dpi=200)
plt.rc("font", size=28)
data = [results_EMNIST["model empirical coverage"],results_EMNIST["CP softmax empirical coverage"], results_EMNIST["CP cumulative empirical coverage"]]

plot_histogram(data, ["r", "y", "b"], ["Cumulative Model Probability", "Normal CP", "Adaptive CP"], "Distribution for Empirical Coverage")
min_x = np.min(data)
max_x = np.max(data)
x = np.arange(int(min_x*test_points), int(max_x*test_points))
a = np.ceil((1-alpha)*(cali_points + 1))
b = np.floor((alpha*(1+cali_points)))
plt.plot(x/test_points, test_points*betabinom.pmf(x, test_points, a, b), '-', linewidth=8, label='betabinom pmf')
            
plt.savefig("./results/EMNIST/empirical coverage")
plt.clf()


# Model accuracy 
# plt.bar(*compute_barplot_data(results_EMNIST["model accuracy"]))
# plt.title("Accuracy of the CNN with one output") 
# plt.savefig("./results/EMNIST/model_accuracy")
# plt.xticks(rotation=45)
# plt.xlabel("CNN accuracy from one class")
# plt.ylabel("Density")
# plt.clf()

# # Emperical coverage
# plt.bar(*compute_barplot_data(results_EMNIST["model empirical coverage"]))
# plt.title("Empirical coverage of CNN with cumulative softmax sets > 1 - alpha") 
# plt.savefig("./results/EMNIST/model_coverage")
# plt.clf()

# # CP softmax emperical coverage
# plt.bar(*compute_barplot_data(results_EMNIST["CP softmax empirical coverage"]))
# plt.title("Empirical coverage of normal CP") 
# plt.savefig("./results/EMNIST/softmax_coverage")
# plt.clf()

# # CP cumulative emperical coverage
# plt.bar(*compute_barplot_data(results_EMNIST["CP cumulative empirical coverage"]))
# plt.title("Empirical coverage of adaptive CP") 
# plt.savefig("./results/EMNIST/adaptive_coverage")
# plt.clf()

# # Prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["model pred sizes"])))
# plt.title("Prediction sizes with cumulative softmax > 1 - alpha") 
# plt.savefig("./results/EMNIST/model_pred_sizes")
# plt.clf()

# # CP softmax prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["CP softmax pred sizes"])))
# plt.title("Prediction sizes with normal CP") 
# plt.savefig("./results/EMNIST/CP_softmax_sizes")
# plt.clf()

# # CP cumulative sum prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["CP cumulative pred sizes"])))
# plt.title("Prediction sizes with adaptive CP") 
# plt.savefig("./results/EMNIST/CP_adaptive_sizes")
# plt.clf()


# Print easily distuingishable statistics 
print(f"CNN average pred size : {np.mean(results_EMNIST['model avg size'])}")
print(f"Normal CP average pred size : {np.mean(results_EMNIST['CP softmax avg size'])}")
print(f"Adaptive CP average pred size : {np.mean(results_EMNIST['CP cumulative avg size'])}")

print(f"CNN empirical coverage : {np.mean(results_EMNIST['model empirical coverage'])}")
print(f"Normal CP empirical coverage : {np.mean(results_EMNIST['CP softmax empirical coverage'])}")
print(f"Adaptive CP empirical coverage : {np.mean(results_EMNIST['CP cumulative empirical coverage'])}")

print(f"CNN average accuracy : {np.mean(results_EMNIST['model accuracy'])}")
#%%