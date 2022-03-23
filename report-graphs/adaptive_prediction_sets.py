#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
plt.rc('axes', axisbelow=True)
plt.rcParams['figure.dpi'] = 300

#%%
# Calculation of score
labels = np.array(["Ice cream", "Wind mill", "Skyscraper", "Dog", "Apple"])
probs = np.array(softmax([5, 4, 3, 2, 1]))
plt.bar(labels[:2], probs[:2], color="tab:blue")
plt.bar(labels[2], probs[2], color="tab:green")
plt.bar(labels[3:], probs[3:], color="tab:grey")
plt.grid()
plt.ylim([0, 1])
plt.xlabel("Classes")
plt.ylabel("Model output")
plt.show()

#%%
# Creating prediction set
labels = np.array(["Dog", "Wind mill", "Apple", "Ice cream", "Skyscraper"])
probs = np.add.accumulate(softmax([10, 9, 6, 6, 5]))

plt.bar(labels[:2], probs[:2], color="tab:blue")
plt.bar(labels[2:], probs[2:], color="tab:grey")
plt.axhline(0.92, color="tab:green", linewidth=3)
plt.grid()
plt.ylim([0, 1])
plt.xlabel("Classes")
plt.ylabel("Model cumulative output")
plt.show()


