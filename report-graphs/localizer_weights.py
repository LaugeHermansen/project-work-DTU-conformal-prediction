#%%
import numpy as np
import matplotlib.pyplot as plt



def plot_weights(x, y, x_0, y_0, weights, title):
    # Set dimensions
    plt.rc("font", size=28)
    plt.figure(figsize=(10, 10), dpi=160)

    # Draw grid
    plt.grid()

    # Plot baseline
    plt.plot(x, y, ".", c="grey", alpha=0.4)
    
    # Plot local point
    plt.plot(x_0, y_0, ".", markersize=16, c="red")
    
    # Shade weights    
    plt.scatter(x, y, c="tab:blue", alpha=weights)
    
    
    # Set plot labels
    plt.title(title)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # Set axis scaling
    plt.axis("equal")
    
    
    # Show
    plt.show()


# Create dataset
rng = np.random.default_rng(16)
n = 1000

x = rng.uniform(0, 1, n)
y = rng.uniform(0, 1, n)

z = np.vstack((x, y)).T
z_0 = np.array([0.5, 0.5])


# Plot KNN weights
k = 100
weights_knn = np.zeros(n)
weights_knn[np.argsort(np.sum((z - z_0)**2, axis=1))[:k]] = 1
plot_weights(x, y, 0.5, 0.5, weights_knn, "KNN, $k = 100$")


# Plot exponential weights
l = 0.2
weights_exp = np.exp(-np.sqrt(np.sum((z - z_0)**2, axis=1))/l)
plot_weights(x, y, 0.5, 0.5, weights_exp, "Exponential, $l = 0.2$")