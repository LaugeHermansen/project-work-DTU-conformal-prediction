
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_protein_data(y_i):

    data = pd.read_csv('data/CASP.csv', header = 0)

    X = data[data.columns[np.setdiff1d(np.arange(len(data.columns)),[y_i])]].to_numpy()

    y = data[data.columns[y_i]].to_numpy().squeeze()

    n_classes = 100
    classes_size = len(y)/n_classes

    stratify = np.empty_like(y)
    mask = np.argsort(y)
    for i in range(n_classes):
        start = np.ceil(classes_size*i).astype(int)
        stop = np.ceil(classes_size*(i+1)).astype(int)
        stratify[mask[start:stop]] = i
    
    return X,y,stratify, data


# if __name__ == "__main__":
#     # test for missing data
#     for c in data.columns:
#         print(f"{c}: nans: {np.sum(np.isnan(data[c]))}, {list(set(data[c].apply(type)))}, [{min(data[c])}, {max(data[c])}]")

#     for i in range(n_classes):
#         plt.hist(y[stratify == i], density=True, bins = 1, color='b')
#     plt.show()
