import numpy as np

def mahalanobis_sqe(length_scale: float):
    def mahalanobis(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = cal_X-x2
            yield np.exp(-np.sum((d @ cov_inv)*d, axis = 1)/(2*length_scale**2))
            
    return mahalanobis

def squared_exponential(length_scale: float):
    def squared_exp(cal_X, test_X):
        for x in test_X:
            yield np.exp(-np.sum((cal_X-x)**2/(2*length_scale**2), axis = 1))

    return squared_exp

def KNN(K):

    def KNN(cal_X, test_X):
        for x2 in test_X:
            d = np.sum((cal_X-x2)**2, axis = 1)
            kernel = np.zeros_like(d)
            for i in np.argsort(d)[:K]:
                kernel[i] = 1
            yield kernel

    # def KNN(cal_X, test_X):
    #     d = np.sum((test_X[:,:,None] - cal_X.T[None,:,:])**2, axis = 1)
    #     rank = np.argsort(d, axis = 1)
    #     kernel = (rank < N).astype(float)
    #     for row in kernel:
    #         yield row
    
    return KNN


def KNN_mahalnobis(N):

    def KNN_mahalnobis(cal_X, test_X):
        cov_inv = np.linalg.pinv(np.cov(cal_X.T))
        for x2 in test_X:
            d = cal_X-x2
            maha = np.sum((d @ cov_inv) * d, axis = 1)
            kernel = np.zeros_like(maha)
            for i in np.argsort(d)[:N]:
                kernel[i] = 1
            yield kernel
            
    return KNN_mahalnobis