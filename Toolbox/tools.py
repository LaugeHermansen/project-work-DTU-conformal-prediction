from sklearn.model_selection import train_test_split
import numpy as np
import sys, inspect
import CP
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


def multiple_split(split_sizes, *arrays, keep_frac = 1.):
    """
    Generalization of train_test_split
    Provide as many splits as desired
    
    Args:
    ----
        split_sizes: list of sizes of the different partitions
        
        *arrays: the arrays to be splitted
        
        keep_rac: the fraction of data that should be kept, float in ]0,1]
        
        keep_frac = 0.7, means that 30% of the data is discarded
    
    Returns:
    --------
        data partitions in the same order as train_test_split
        so if arrays = X,y and splits are 0.3,0.2,0.5
        it returns X1,X2,X3,y1,y2,y3
    """
    if not 0 < keep_frac <= 1: raise ValueError(f"keep_pct must be in range ]0,1] - was {keep_frac}")
    
    if keep_frac < 1:
        temp = train_test_split(*arrays, stratify=arrays[-1], test_size=1-keep_frac)
        arrays = [temp[i*2] for i in range(len(arrays))]

    split_sizes = np.array(split_sizes)
    splits_sum = np.cumsum(split_sizes[::-1])[::-1]
    if splits_sum[0] != 1: print("Warning: splits should sum to 1")
    test_sizes = 1-split_sizes/splits_sum
    ret = []
    for test_size in test_sizes[:-1]:
        temp = train_test_split(*arrays, stratify=arrays[-1], test_size=test_size)
        arrays_ret = [temp[i*2  ] for i in range(len(arrays))]
        arrays =     [temp[i*2+1] for i in range(len(arrays))]
        ret.append(arrays_ret)
    ret.append(arrays)
    return tuple([partition_of_variabel for variable in zip(*ret) for partition_of_variabel in variable])



def get_all_cp_models(locals):
    """
    collect all cp_models in a list
    a call to this function must look exactly like
    `get_all_cp_models(locals())`

    Args:
    -----
        locals:
            the local variables of the main script.
            can be accessed with locals()
    
    Returns:
    --------
        cp_models:
            a list object containing all the CP models in locals
            (which means the main script if the function was called with
            `locals=locals()` )
    """
    return [b for b in locals.values() if inspect.isclass(b.__class__) and not inspect.isclass(b) and '<CP.' == repr(b)[:4]]


def stratified_coverage(feature_samples: np.ndarray, in_pred_set: np.ndarray, n_features: int = None):

    """
    Compute either FSC or MSC, depending on the input, `feature_samples`
    It returns all sc's and not just the minimum.

    Args:
    -----
        feature_samples:
            the feature to stratify over.
            for MSC, pass `feature_samples=pred_set_sizes`

        in_pred_set:
            the output from `cp_model.evaluate_coverage.in_pred_set`
        
        n_features:
            the number of features in which to partition the 
            data if `feature_samples` is continuous.
            if `None`, the unique values in `feature_samples`
            will be used

    Returns:
    -------
        labels:
            a list of unique labels, 1d array of length |N_labels|

        empirical_conditional_coverage:
            the empicial coverages conditioned on 
            the labels in `labels`.

        n_conditional_samples:
            the number of samples in test set when conditioned
            on the labels in `labels`.
    """
    temp_feature_coverage = defaultdict(list)

    if n_features != None:
        label_lookup = {}
        classes_size = len(feature_samples)/n_features
        label_samples = np.zeros(len(feature_samples)).astype(str)
        mask = np.argsort(feature_samples)
        for label in range(n_features):
            start = np.ceil(classes_size*label).astype(int)
            stop = np.ceil(classes_size*(label+1)).astype(int)
            label_samples[mask[start:stop]] = str(label)
            label_lookup[str(label)] = f"({feature_samples[mask[start]]:.3} ,  {feature_samples[mask[min(stop, len(mask)-1)]]:.3}("
    else:
        label_samples = feature_samples.copy().astype(str)
        label_lookup = {i:i for i in set(label_samples)}

    for label, ips in zip(label_samples, in_pred_set):
        temp_feature_coverage[label].append(ips)

    labels = np.zeros(len(temp_feature_coverage)).astype(str)
    empirical_conditional_coverage = np.zeros(len(labels))
    n_conditional_samples = np.zeros(len(labels)).astype(int)

    for i, (label, ips_list) in enumerate(temp_feature_coverage.items()):
        labels[i] = str(label_lookup[label])
        empirical_conditional_coverage[i] = np.mean(ips_list)
        n_conditional_samples[i] = len(ips_list)
    return labels, empirical_conditional_coverage, n_conditional_samples


