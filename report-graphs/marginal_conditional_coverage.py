#%%
from matplotlib import pyplot as plt
import numpy as np


#%%
rng = np.random.default_rng(2)
n = 200
p = 0.8
c0 = 0.8
c1 = 0.1

g = np.array(rng.uniform(0, 1, n) < p, dtype=int)
x = rng.normal(0, 0.2, n) + g
y = rng.normal(0, 0.2, n)

d = np.zeros_like(x, dtype=bool)
d[g == 0] = rng.binomial(1, c0, np.sum(g == 0))
d[g == 1] = rng.binomial(1, c1, np.sum(g == 1))



#%%
# Plot marginally
plt.plot(x[d == 0] - g[d == 0], y[d == 0], ".", c="green")
plt.plot(x[d == 1] - g[d == 1], y[d == 1], ".", c="red")
plt.text(0, -0.7, "All patients", ha="center")



plt.axis("off")
plt.legend(["Healthy", "Diseased"])
plt.show()


# Plot group wise
plt.plot(x[d == 0], y[d == 0], ".", c="green")
plt.plot(x[d == 1], y[d == 1], ".", c="red")
plt.plot([0.5, 0.5], [-1, 1], "--", c="black")
plt.text(0, -0.8, "Group 1", ha="center")
plt.text(1, -0.8, "Group 2", ha="center")

plt.axis("off")
plt.legend(["Healthy", "Diseased"])
plt.show()

