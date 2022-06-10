
import numpy as np
a = np.array([[0,1],[1,0],[1,0],[1,1]]).astype(bool)
y = np.array([0,0,1,1])
print(a[np.arange(len(y)), y])

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



