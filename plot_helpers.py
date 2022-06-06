import numpy as np
import matplotlib.pyplot as plt

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