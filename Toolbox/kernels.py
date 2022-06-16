import numpy as np

def mahalanobis_exponential(length_scale: float):

    def mahalanobis_exponential(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = (cal_X-x2).astype(np.float64)
            kernel = np.exp(-np.sqrt(np.sum((d @ cov_inv)*d, axis = 1))/length_scale)
            yield kernel

    return mahalanobis_exponential

def exponential(length_scale: float):
    def exponential(cal_X, test_X):
        for x2 in test_X:
            d = (cal_X-x2).astype(np.float64)
            kernel = np.exp(-np.sqrt(np.sum(d**2, axis = 1))/length_scale)
            yield kernel

    return exponential

def mahalanobis_squared_exponential(length_scale: float):

    def mahalanobis_squared_exponential(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = (cal_X-x2).astype(np.float64)
            kernel = np.exp(-np.sum((d @ cov_inv)*d, axis = 1)/length_scale)
            yield kernel

    return mahalanobis_squared_exponential

def squared_exponential(length_scale: float):
    def squared_exponential(cal_X, test_X):
        for x2 in test_X:
            d = (cal_X-x2).astype(np.float64)
            kernel = np.exp(-np.sum(d**2, axis = 1)/length_scale)
            yield kernel

    return squared_exponential

def KNN(N):

    def KNN(cal_X, test_X):
        for x2 in test_X:
            d = np.sum((cal_X-x2)**2, axis = 1)
            kernel = np.zeros_like(d)
            for i in np.argsort(d)[:N]:
                kernel[i] = 1
            yield kernel
    
    return KNN


def mahalanobis_KNN(N):

    def mahalanobis_KNN(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = cal_X-x2
            maha = np.sum((d @ cov_inv) * d, axis = 1)
            kernel = np.zeros_like(maha)
            for i in np.argsort(maha)[:N]:
                kernel[i] = 1
            yield kernel
            
    return mahalanobis_KNN