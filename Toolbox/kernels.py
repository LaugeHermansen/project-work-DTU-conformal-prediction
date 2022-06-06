import numpy as np

def mahalanobis_sqe(X1, X2):
    cov_inv = np.linalg.pinv(np.cov(X1.T))
    for x2 in X2:
        d = X1-x2
        yield np.exp(-np.sum((d @ cov_inv)*d, axis = 1)/0.5)


def squared_exponential(X1, X2):
    for x in X2:
        yield np.exp(-np.sum((X1-x)**2/0.5, axis = 1))
