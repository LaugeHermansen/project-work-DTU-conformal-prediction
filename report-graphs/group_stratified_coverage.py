#%%
from matplotlib import pyplot as plt
import numpy as np


#%%
rng = np.random.default_rng(16)
n = 200
p = np.array([0.15, 0.4, 0.15, 0.2, 0.1])
c = np.array([0.85, 0.85, 0.80, 0.50, 0.8])


g = rng.multinomial(1, p, n).argmax(axis=1)

x = rng.normal(0, 0.2, n)
y = rng.normal(0, 0.2, n)

d = np.zeros_like(x, dtype=int)
for i in range(len(p)):
    d[g == i] = rng.binomial(1, c[i], np.sum(g == i))



#%%
plt.rc("font", size=14)
plt.figure(figsize=(10, 6), dpi=200)

plt.scatter(x[d == 0], y[d == 0], s=6, alpha=0.6, c="red")
plt.scatter(x[d == 1], y[d == 1], s=6, alpha=0.6, c="green")
plt.text(0, 0.8, f"$\hat U = {np.mean(d)*100:.2f}\%$", size=20, ha="center")
plt.arrow(0, -0.8, 0, -0.4, width=0.04, color="black")


group_y_offset = -2.2
group_scale = 1
group_separation_scale = 1.6
group_x_offset = -(len(p)-1)/2 * group_separation_scale

x_g = x * group_scale + g*group_separation_scale + group_x_offset
y_g = y * group_scale + group_y_offset
plt.scatter(x_g[d == 0], y_g[d == 0], s=6, alpha=0.6, c="red")
plt.scatter(x_g[d == 1], y_g[d == 1], s=6, alpha=0.6, c="green")

for i in range(len(p)):
    plt.text(
        i * group_separation_scale + group_x_offset, 
        -3.2 - (i % 2) * 0.6, 
        f"$\hat U_{i+1} = {np.mean(d[g == i])*100:.2f}\%$", 
        size=16,
        ha="center",
        color = "black" if i != 3 else "blue"
    )
    
plt.arrow(
    3 * group_separation_scale + group_x_offset, 
    -4, 
    0, 
    -0.4, 
    width=0.04, 
    color="black"
)
plt.text(
    3 * group_separation_scale + group_x_offset, 
    -5.1,
    f"$\hat U_C = {np.mean(d[g == 3])*100:.2f}\%$", 
    size=20,
    ha="center"
)

plt.axis("equal")
plt.axis("off")
plt.legend(["Wrong", "Correct"])
plt.show()
