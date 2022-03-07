import torch
import torch.nn
from model import CNN_class
from lauges_tqdm import tqdm
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from typing import Tuple


def load_data(batch_size = 50, data_workers = 1, test_size_train = 0.4, test_size_test = 0.5) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])

    training_set = datasets.EMNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    test_set = datasets.EMNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)

    _, train_indices = train_test_split(np.arange(len(training_set)), test_size=test_size_train, stratify=training_set.targets)
    training_set = data_utils.Subset(training_set, train_indices)

    val_indices, test_indices = train_test_split(np.arange(len(test_set)), test_size=test_size_test, stratify=test_set.targets)
    validation_set = data_utils.Subset(test_set, val_indices)
    test_set = data_utils.Subset(test_set, test_indices)

    train_dl = DataLoader(training_set, batch_size=batch_size, num_workers=data_workers)
    validation_dl = DataLoader(validation_set, batch_size=batch_size, num_workers=data_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=data_workers)

    return train_dl, validation_dl, test_dl

def score_function(output: torch.Tensor,y_true: torch.Tensor = None) -> torch.Tensor:
    """
    calculate score function for n data points

    Parameters:
    -----------
    -    output: (n,m) torch tensor array - the softmax output
    -    y_true: (n,) torch tensor - true categories - zero indexed

    Return:
    -------
    -    (n,) tensor - score function
    """
    if y_true == None: return output
    else:              return output[:,y_true]

def get_quantile(model, calibration_dl, alpha, score_function) -> float:
    """
    get the empirical 1-alpha quantile of calibration data

    Parameters:
    ----------
        - calibration_dl: torch Dataloader object
        - alpha: significance level of prediction set
    
    Return:
    -------
        - the empirical 1-alpha quantile of calibration data
    """
    s = torch.vstack(*[score_function(model(x),y) for (x,y) in calibration_dl])
    n = len(s)
    q = np.ceil((n+1)(1-alpha))/n
    return np.quantile(s, q)


def predict(model, validation_dl, quantile):
    """get prediciton set

    Parameters:
    ----------
    - model: you know
    - test_dl: test set torch data loader
    - quantile: 1-alpha quantile

    Return:
    -------
    - prediction set: for each data point
    - y_true: the true labels
    """
    prediction_set = []
    y_true = []
    for (x,y) in validation_dl:
        y_out = model(x)
        prediction_set.append(y_out[y_out <= quantile])
        y_true.append(y)
    prediction_set = torch.vstack(*prediction_set)
    y_true = torch.vstack(*y_true)
    return prediction_set, y_true

def evaluate_coverage(prediction_sets,y_true):
    """
    estimated coverage of one trial

    Parameters:
    ------
    - prediction_sets: (n,m) np array
    - y_true: (n,)

    Return:
    -------
    C_j: estimated coverage

    """
    return np.mean(map(lambda x: x[1] in x[0], zip(y_true, prediction_sets)))

if __name__ == '__main__':
    
    validation_accuracies = np.load('bayesian_optimization_accuracies_val.npy')
    hyperparameters = np.load('bayesian_optimization_hyperparameters.npy')
    hyperparameter = hyperparameters[np.argmax(validation_accuracies)]

    _, calibration_dl, validation_dl = load_data()

    model = CNN_class(*hyperparameter[2:])
    model.load_state_dict(torch.load('bayesian_optimization_best_model.pt'))

    test_image = next(iter(validation_dl))
    alpha = 0.05
    quantile = get_quantile(model,calibration_dl,alpha,score_function)
    prediction_set,y_true = predict(model, validation_dl, quantile)
    coverage = evaluate_coverage(prediction_set,y_true)

    print(coverage)



