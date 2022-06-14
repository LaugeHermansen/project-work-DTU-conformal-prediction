import numpy as np

def binary_search(cdf, i=0, j=None):
    """
    return the index of the first score where
    cdf >= self.alpha using binary search
    """
    j = len(cdf) if j == None else j
    m = int((i+j)/2)
    if i == j:  return i
    if cdf[m] < 1-alpha:   return binary_search(cdf, m+1, j)
    elif cdf[m] > 1-alpha: return binary_search(cdf, i, m)
    else: return m

def baseline(cdf):
    for i,v in enumerate(cdf):
        if v >= 1-alpha:
            return i
    return len(cdf) + 1

alpha = 0.19


cdf_raw = np.random.uniform(0, 0.8, 100000)
cdf = np.sort(cdf_raw)

print(baseline(cdf))
print(binary_search(cdf))
