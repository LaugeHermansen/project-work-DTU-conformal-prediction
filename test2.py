#%%
from Toolbox.kernels import squared_exponential
import numpy as np

A = np.array([[1,1],[2,2],[4,2]])

B1 = np.array([[3,6],
              [2,2],
              [1,6],
              [7,8],
              ])
B2 = np.array([[9,6],
              [2,9],
              [6,6],
              [7,12],
              ])

B = np.stack((B1,B2))


def squared_exponential2(length_scale: float):
    def squared_exp(cal_X, test_X):
        d_sq = np.sum((test_X[:,:,None] - cal_X.T[None,:,:])**2, axis = 1)
        sqex = np.exp(-d_sq/(2*length_scale**2))
        for row in sqex:
            yield row

    return squared_exp

sq1 = squared_exponential(1)
sq2 = squared_exponential2(1)

for w1, w2 in zip(sq1(A,B1), sq2(A,B1)):
    print(np.all(w1==w2))
    print(w1,w2)


#%%



# d = B1[:,:,None] - A.T[None,:,:]

# length_scale = 1/2

# def mahalanobis(cal_X, test_X):
#     cov_inv = np.linalg.pinv(np.cov(cal_X.T))
#     d = cal_X[:,:,None] - test_X.T[None,:,:]
#     return np.exp(-np.sum((d @ cov_inv)*d, axis = 1)/(2*length_scale**2))


# #%%

# class CP:

#     def __init__(self, name, lm):
#         if hasattr(lm, "__call__"): pass
#         else: lm.__class__.__call__ = getattr(lm.__class__, 'predict') 
#         self.name = name
#         self.lm = lm
#     def pred(self):
#         return self.lm()


# class øv:
#     def __init__(self):
#         self.n = None

#     def fit(self, n):
#         self.n = n
    
#     def predict(self):
#         return self.n


# lm1 = øv()
# lm2 = øv()

# lm1.fit(4)
# lm2.fit(7)

# cp1 = CP(3, lm1)
# cp2 = CP(7, lm2)

# print(lm1.predict(),
# lm1(),
# lm2.predict(),
# lm2())



