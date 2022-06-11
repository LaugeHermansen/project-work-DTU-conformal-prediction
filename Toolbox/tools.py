from sklearn.model_selection import train_test_split
import numpy as np
import sys, inspect
import CP
import matplotlib.pyplot as plt


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
    return [b for b in locals.values() if inspect.isclass(b.__class__) and not inspect.isclass(b) and 'CP.' in repr(b)]
