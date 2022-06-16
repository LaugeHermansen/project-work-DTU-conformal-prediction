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
    
    return np.array(attributes)[mask], np.array(heights)[mask]


def barplot(heights, labels, figsize = None):

    n_labels = len(labels)

    barWidth = 1/(n_labels + 1)
    fig, ax = plt.subplots(figsize = figsize)
    for i, label, height in zip(range(n_labels), labels, heights):
        plt.bar(label + barWidth*i, height, width = barWidth, label = label)
    plt.xticks(label + barWidth*(n_labels-1)/2, label, ha="right")
    return fig, ax

def scatter(axis_data, color_data, feature1: int = 0, feature2: int = 1,
           alpha: float = 0.1, adapt_lim_level = 1.0, adapt_lim_n_std = 0.5, s = 3):

    plt.scatter(axis_data[:,feature1],axis_data[:,feature2], edgecolors='none', s=s, alpha=alpha,c=color_data)
    plt.colorbar()
    if adapt_lim_level != 1.0: plt.clim(compute_lim(color_data, adapt_lim_level, adapt_lim_n_std))

def compute_lim(array, level = 0.005, n_std = 0.5):
    return np.quantile(array, [level/2,1-level/2]) + n_std*np.std(array)*np.array([-1,1])
