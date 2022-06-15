#%%

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def v(x_grid):
    return np.sin(x_grid)*x_grid + np.random.rand(len(x_grid))*x_grid**2/4

def h(x, x_grid):
    return np.max(np.vstack((1 - np.abs(x-x_grid), np.zeros_like(x_grid))), axis = 0)

def mean(h_out,v_out):
    return np.sum(h_out*v_out)/np.sum(h_out)


x0 = 7
v_est = []
for i in range(2, 6):
    x_grid = np.linspace(4,10,10**i)
    v_est.append(np.array([mean(h(x0,x_grid), v(x_grid)) for i in tqdm(range(1000))]))
    plt.hist(v_est[-1], bins = 40)

plt.show()
# plt.plot(x_grid, v(x_grid), '.')