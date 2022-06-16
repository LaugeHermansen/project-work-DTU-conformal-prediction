#%%
import numpy as np
import pandas as pd 
from scipy.stats import norm
import os 
import pickle
import matplotlib.pyplot as plt
from sympy import hyper

from CP import ClassificationCumulativeSoftmax, ClassificationSoftmax
from CP.CP_base import CPEvalData

from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from Toolbox.tools import multiple_split
#from Toolbox.kernels import mahalanobis_exponential, exponential

from Toolbox.kernels import mahalanobis_KNN, KNN, exponential
from Toolbox.plot_helpers import barplot, compute_barplot_data
from Toolbox.tools import get_all_cp_models, multiple_split
#plt.rcParams['text.usetex'] = True

class Classification_Wrapper():
    # makes the predict_proba method the call methods so it can be used with the rest of the framework
    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        
    def __call__(self, X):
        return self.model.predict_proba(X)

class Coverage_Wrapper(): 
    # Makes it possible to evaluate models without applying CP 
    def __init__(self, model):
        self.model = model 
        self.alpha = self.model.alpha 

    def __call__(self, X):
        predictions = self.model(X)
        indices = np.argsort(-predictions, axis=1)
        reverse_indices = np.argsort(indices, axis=1)
        sorted = np.take_along_axis(predictions, indices, axis=1)
        weighted_preds = np.cumsum(sorted, axis = 1)
        weighted_preds = np.take_along_axis(weighted_preds, reverse_indices, axis=1)
        
        pred = np.argmin(weighted_preds, axis=1)
        preds = weighted_preds < 1-alpha
        weighted_preds[preds] += 1
        preds[np.arange(len(preds)), np.argmin(weighted_preds, axis=1)] = 1
        
        return pred, preds
        
    def eval_coverage(self, X, y):
        pred, preds = self(X) 
        pred_sizes = np.sum(preds, axis=1)
        avg_size = np.mean(pred_sizes)

        accuracy = np.mean(pred == y) 
        empirical_covarage = np.mean([preds[i][y[i]] for i in range(len(y))])
        return pred_sizes, avg_size, accuracy, empirical_covarage


if not "results" in os.listdir("."):
    os.mkdir("./results")
if not "adult" in os.listdir("./results"):
    os.mkdir("./results/adult ")

alpha = 0.1
#%% create data sets and remove missing/faulty entries 
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'above-50']
one_hots = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'above-50']
label = "education"

#Load data
data = pd.read_csv("data/adult.data", names = columns).applymap(lambda x: x.strip() if isinstance(x, str) else x)
n1=len(data)
#remove missing values
data = data.applymap(lambda x: np.nan if x == "?" else x).dropna(axis = 0)
data = data.applymap(lambda x: np.nan if x == "education" else x).dropna(axis = 0)
n2=len(data)

print(f"Amount of removed/faulty data points {n1-n2}")
data["race"].value_counts()
n_white = data["race"].value_counts()["White"]
print(f"Total amount of people {n2}, Non-white people {n2-n_white} which is {(n2 - n_white)/n2}")

#%% Preprocessing 
# One-hot encode and split into train and test
#split X and y
data_X = data[np.setdiff1d(columns,[label])]
data_y = data[[label]]

#One hot encode X
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(data_X[one_hots])
encodings = enc.transform(data_X[one_hots])

X = data_X.drop(labels = one_hots, axis = 1).to_numpy()
X = np.hstack((X, encodings.toarray()))

#Label encode y
y = data_y.to_numpy().squeeze()
y_predicate = (y=="9th") | (y=="10th") | (y=="11th") | (y=="12th")
y[y_predicate] = "9th-12th"
y_predicate = (y=="5th-6th") | (y=="7th-8th")
y[y_predicate] = "5th-8th"

lenc = LabelEncoder()
lenc.fit(y)
y = lenc.transform(y)

#Split data sets (train, cal, test)
train_X, cal_X, test_X, train_y, cal_y, test_y = multiple_split((0.2,0.3,0.5), X,y, keep_frac = 0.3)


#%% load models 
model = RandomForestClassifier(n_estimators=500, n_jobs = 6)
model.fit(train_X, train_y)
print(model.score(train_X, train_y))
print(model.score(test_X, test_y))


results_adult = {"model pred sizes" : [],
                  "model avg size" : [], 
                  "model accuracy" : [],
                  "model empirical coverage" : [], 
                  "CP softmax pred sizes" : [],
                  "CP cumulative pred sizes" : [], 
                  "CP softmax avg size" : [], 
                  "CP cumulative avg size" : [], 
                  "CP softmax empirical coverage" : [],
                  "CP cumulative empirical coverage" : [] 
}

# Create evaluation model 
evaluate_model = Coverage_Wrapper(model = Classification_Wrapper(model, alpha))

# Apply CP framework 
CP_softmax = ClassificationSoftmax(Classification_Wrapper(model, alpha), cal_X, cal_y, alpha)
CP_cumulative = ClassificationCumulativeSoftmax(Classification_Wrapper(model, alpha), cal_X, cal_y, alpha)

# evaluate model and CP models 
model_pred_sizes, model_avg_size, model_accuracy, model_empirical_covarage = evaluate_model.eval_coverage(test_X, test_y) 
softmax_results = CP_softmax.evaluate_coverage(test_X, test_y) 
cumulative_results = CP_cumulative.evaluate_coverage(test_X, test_y)

# Save configurations 
results_adult["model pred sizes"].append(model_pred_sizes)
results_adult["model avg size"].append(model_avg_size)
results_adult["model accuracy"].append(model_accuracy)
results_adult["model empirical coverage"].append(model_empirical_covarage)
results_adult["CP softmax pred sizes"].append(softmax_results.pred_set_sizes)
results_adult["CP cumulative pred sizes"].append(cumulative_results.pred_set_sizes)
results_adult["CP softmax avg size"].append(np.mean(softmax_results.pred_set_sizes))
results_adult["CP cumulative avg size"].append(np.mean(cumulative_results.pred_set_sizes))
results_adult["CP softmax empirical coverage"].append(softmax_results.empirical_coverage)
results_adult["CP cumulative empirical coverage"].append(cumulative_results.empirical_coverage)


results_adult_file = open("./results/adult/results_adult.pkl", "wb")
pickle.dump(results_adult, results_adult_file)
results_adult_file.close()
#%%
# Show the results 

results_adult_file = open("./results/adult/results_adult.pkl", "rb")
output = pickle.load(results_adult_file)

# # Prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_adult["model pred sizes"])))
# plt.title("Prediction sizes with cumulative softmax > 1 - alpha") 
# plt.savefig("./results/adult/model_pred_sizes")
# plt.clf()

# # CP softmax prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_adult["CP softmax pred sizes"])))
# plt.title("Prediction sizes with normal CP") 
# plt.savefig("./results/adult/CP_softmax_sizes")
# plt.clf()

# # CP cumulative sum prediction sizes 
# plt.bar(*compute_barplot_data(np.hstack(results_adult["CP cumulative pred sizes"])))
# plt.title("Prediction sizes with adaptive CP") 
# plt.savefig("./results/adult/CP_adaptive_sizes")
# plt.clf()

# All three prediction sets in one plot 
plt.bar(*compute_barplot_data(np.hstack(results_adult["model pred sizes"])), alpha=0.5, legend="Cumulative softmax > 1 - alpha")

# CP softmax prediction sizes 
plt.bar(*compute_barplot_data(np.hstack(results_adult["CP softmax pred sizes"])), alpha=0.5, legend="Normal CP")

# CP cumulative sum prediction sizes 
plt.bar(*compute_barplot_data(np.hstack(results_adult["CP cumulative pred sizes"])), alpha=0.5, legend="Adaptive CP")
plt.title("Prediction set sizes") 
plt.savefig("./results/adult/pred_set_sizes")
plt.legend()
plt.clf()

# Print easily distuingishable statistics 
print(f"Random Forest average pred size : {np.mean(results_adult['model avg size'])}")
print(f"Normal CP average pred size : {np.mean(results_adult['CP softmax avg size'])}")
print(f"Adaptive CP average pred size : {np.mean(results_adult['CP cumulative avg size'])}")

print(f"Random Forest empirical coverage : {np.mean(results_adult['model empirical coverage'])}")
print(f"Normal CP empirical coverage : {np.mean(results_adult['CP softmax empirical coverage'])}")
print(f"Adaptive CP empirical coverage : {np.mean(results_adult['CP cumulative empirical coverage'])}")

# %%
