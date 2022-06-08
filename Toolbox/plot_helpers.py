import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_barplot_data(array):
    temp = {}
    for a in array:
        if a in temp: temp[a] += 1
        else: temp[a] = 1
    attributes, heights = tuple(zip(*list(temp.items())))
    mask = np.argsort(attributes)
    
    return np.array(attributes)[mask].astype(str), np.array(heights)[mask]


def barplot(xtick_labels, heights, labels, figsize = None):
    if not len(labels) == len(heights):
        raise ValueError('Number of labels must equal number of height arrays')

    xticks = np.arange(len(xtick_labels))
    n_labels = len(labels)

    barWidth = 1/(n_labels + 1)
    fig, ax = plt.subplots(figsize = figsize)
    for i, label, height in zip(range(n_labels), labels, heights):
        plt.bar(xticks + barWidth*i, height, width = barWidth, label = label)
    plt.xticks(xticks + barWidth*(n_labels-1)/2, xtick_labels, ha="right")
    return fig, ax

def scatter_y_on_pcs(X_pca, y, pc1: int = 0, pc2: int = 1, ax: plt.Axes = None):

    ax_ = plt if ax == None else ax
    ax_.scatter(X_pca[:,pc1],X_pca[:,pc2], edgecolors='none', s=3, alpha=0.1,c=y)
    ax_.colorbar()
    ax_.title('y on first two PCs')
    if ax == None: plt.show()
    