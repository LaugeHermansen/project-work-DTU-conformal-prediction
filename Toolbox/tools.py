from sklearn.model_selection import train_test_split
import numpy as np
import inspect
from CP import CPEvalData
import matplotlib.pyplot as plt
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import pandas as pd
from dataclasses import dataclass
import os

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
    if splits_sum[0] < 1-1e-4: print("Warning: splits should sum to 1")
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


def evaluate(X, in_pred_set, bins=20, min_x=0, max_x=1):
    in_pred = np.zeros(bins)
    count = np.zeros(bins)
    for i in range(len(X)): 
        index = int((X[i] - min_x)//((max_x - min_x)/bins))
        in_pred[index] += in_pred_set[i]
        count[index] += 1 
    return in_pred/count

def evaluate2(X, in_pred_set, bins=20, min_x=0, max_x=1):
    in_pred = np.zeros(bins)
    count = np.zeros(bins)
    for i in range(len(X)): 
        index = int((X[i] - min_x)//((max_x - min_x)/bins))
        in_pred[index] += in_pred_set[i]
        count[index] += 1 
    return in_pred/count, count



class GPWrapper(GaussianProcessRegressor): 
    
    def predict(self, X): 
        y_mean, y_std = super().predict(X, return_std=True)
        pred_interval = y_mean[:, None] + np.array([-1, 0, 1]) * y_std[:,None] * norm.ppf(q = 1-self.alpha)
        return pred_interval
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

@dataclass
class CPRegressionResults:
    cp_results: CPEvalData
    data: pd.DataFrame
    X_standard: np.ndarray
    y: np.ndarray
    stratify: np.ndarray
    train_X: np.ndarray
    cal_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    cal_y: np.ndarray
    test_y: np.ndarray
    train_strat: np.ndarray
    cal_strat: np.ndarray
    test_strat: np.ndarray

    def save(self, path):
        pd.to_pickle(self, path)
    
    def load(path):
        ret = pd.read_pickle(path)
        if ret.__class__ == CPRegressionResults: return ret
        else: raise FileExistsError('File is not a CPRegressionResults object')
     

def mpath(path):
    path = path.strip('/')
    path_list = path.split('/')
    for i, p in enumerate(path_list):
        root = "./" + "/".join(path_list[:i])
        if not p in os.listdir(root):
            os.mkdir(root + "/" + p)
    return path + "/"