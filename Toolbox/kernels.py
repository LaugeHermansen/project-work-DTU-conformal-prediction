import numpy as np

def mahalanobis_exponential(length_scale: float):

    def mahalanobis_exponential(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = cal_X-x2
            kernel = np.exp(-np.sqrt(np.sum((d @ cov_inv)*d, axis = 1))/length_scale)
            yield kernel

    return mahalanobis_exponential

def exponential(length_scale: float):
    def exponential(cal_X, test_X):
        for x in test_X:
            kernel = np.exp(-np.sqrt(np.sum((cal_X-x)**2, axis = 1))/length_scale)
            yield kernel

    return exponential

def KNN(N):

    def KNN(cal_X, test_X):
        for x2 in test_X:
            d = np.sum((cal_X-x2)**2, axis = 1)
            kernel = np.zeros_like(d)
            for i in np.argsort(d)[:N]:
                kernel[i] = 1
            yield kernel
    
    return KNN


def KNN_mahalnobis(N):

    def KNN_mahalnobis(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = cal_X-x2
            maha = np.sum((d @ cov_inv) * d, axis = 1)
            kernel = np.zeros_like(maha)
            for i in np.argsort(maha)[:N]:
                kernel[i] = 1
            yield kernel
            
    return KNN_mahalnobis