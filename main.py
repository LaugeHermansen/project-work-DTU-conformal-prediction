#form conformal_prediction import CP_softmax
from CP import Classification_cumulative_softmax
import torch
import torch.nn
from model import CNN_class
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
from matplotlib import pyplot as plt


def load_data(batch_size = 500, data_workers = 1, test_size_train = 0.4, test_size_test = 0.5) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()]) # transforms.LinearTransformation(torch.Tensor([[1, 0, 0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]), torch.Tensor

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


def plot_letters(n, dl):
    batch = next(iter(dl))
    imgs = batch[0].detach().numpy()

    fig, axs = plt.subplots(1, n)
    fig.suptitle('Handwritten digits')
    for i in range(n):
        plt.sca(axs[i])
        plt.imshow(imgs[i].reshape(28, 28).T, interpolation='nearest')

    plt.show()



if __name__ == '__main__':
    validation_accuracies = np.load('bayesian_optimization_accuracies_val.npy')
    hyperparameters = np.load('bayesian_optimization_hyperparameters.npy')
    hyperparameter = hyperparameters[np.argmax(validation_accuracies)]

    _, calibration_dl, validation_dl = load_data()

    plot_letters(6, calibration_dl)

    model = CNN_class(*hyperparameter[2:])
    print(hyperparameter) 
    model.load_state_dict(torch.load('bayesian_optimization_best_model.pt'))

    model1 = lambda x: torch.nn.functional.softmax(model(torch.permute(x, (0,1,3,2))), dim=1).detach().numpy()

    accuracies = []
    for (x_cal,y_cal) in tqdm(calibration_dl):
        CP_model = Classification_cumulative_softmax(model=model1,
                                                              calibration_set_x=x_cal,
                                                              calibration_set_y=y_cal.detach().numpy(),
                                                              alpha=0.05)
        temp_acc =[]
        for (x_val, y_val) in validation_dl:
            conf_intervals = CP_model.predict(x_val)

            acc = np.mean([y_hat[y] for y, y_hat in zip(y_val.detach().numpy(), conf_intervals)])
            temp_acc.append(acc)
        accuracies.append(temp_acc)

    print(accuracies)



