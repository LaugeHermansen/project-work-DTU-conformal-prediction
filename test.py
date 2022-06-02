import numpy as np

def binary_search(cdf, alpha, i=0, j=None):
    j = len(cdf) if j == None else j
    m = int((i+j)/2)
    # print(i,m,j)
    if i == j:  return i
    if cdf[m] < alpha:   return binary_search(cdf, alpha, m+1, j)
    elif cdf[m] > alpha: return binary_search(cdf, alpha, i, m)
    else: return m
    


w = np.array([1,9,5,12,3,10])
s = np.array([1,2,3,4,5,6])

wc = np.cumsum(w)

cdf = wc/wc[-1]

level = 0.99

print(cdf)
print(np.argmax(cdf >= level))
print(binary_search(cdf, level))

print(wc)


