import numpy as np

def mahalanobis_sqe(length_scale: float):
    def mahalanobis(X1, X2):
        cov_inv = np.linalg.pinv(np.cov(X1.T))
        for x2 in X2:
            d = X1-x2
            yield np.exp(-np.sum((d @ cov_inv)*d, axis = 1)/(2*length_scale**2))
            
    return mahalanobis


def squared_exponential(length_scale: float):
    def squared_exp(X1, X2):
        for x in X2:
            yield np.exp(-np.sum((X1-x)**2/(2*length_scale**2), axis = 1))

    return squared_exp