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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#from Toolbox.tools import multiple_split
#from Toolbox.kernels import mahalanobis_exponential, exponential

from Toolbox.kernels import mahalanobis_KNN, KNN, exponential
from Toolbox.plot_helpers import barplot, compute_barplot_data
from Toolbox.tools import get_all_cp_models, multiple_split
#plt.rcParams['text.usetex'] = True


if not "results" in os.listdir("."):
    os.mkdir("./results")
if not "ADULT" in os.listdir("./results"):
    os.mkdir("./results/ADULT ")

#%% create data sets 
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'above-50']
one_hots = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'above-50']
label = "education"

#Load data
data = pd.read_csv("data/adult.data", names = columns).applymap(lambda x: x.strip() if isinstance(x, str) else x)
n1=len(data)
#remove missing values
data = data.applymap(lambda x: np.nan if x == "?" else x).dropna(axis = 0)
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

# TODO can you apply Coverage_Wrapper directly to any classificaition model from sklearn? 
# Nope gotta make a wrapper that makes predict_proba the call method, but then it should be good 
class CNN():
    # Loads CNN, trains it if it hasn't been and implements a call method 
    def __init__(self, width, depth, lr=None, wd=None, train_dl=None, params=None, alpha=0.9):
        self.model = CNN_class(width, depth)
        self.alpha = alpha

        if params != None: 
            self.model.load_state_dict(params)
        else: 
            train(self.model, train_dl, lr, wd)
        
    def __call__(self, X):
        predictions = torch.nn.functional.softmax(self.model(torch.permute(X, (0,1,3,2))), dim=1).detach().numpy()
        return predictions

class Coverage_Wrapper(): 
    # Makes it possible to evaluate models without applying CP 
    def __init__(self, model):
        self.model = model 
        self.alpha = self.model.alpha 

    def __call__(self, X):
        predictions = model(X)
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


# Hyper parameters 
fig_size = (13, 8)
alpha = 0.05
#kernels = [None, exponential(0.01)]
#kernel_names = ["Normal CP", "Naive LCP Exponential"]

#colors = ["k", "r", "b", "g", "m", "y"] 

size = 2000
train_points = 1000
cali_points = 500
test_points = size - train_points - cali_points
train_cali_test = np.array([train_points, cali_points, test_points])/size


#%% Show one instance of the data set 
train_dl, cali_dl, test_dl = load_data(train_points=train_points, cali_points=cali_points, test_points=test_points)
#plt.rcParams["figure.figsize"] = (10, 5) # TODO fix so that there is not all that white space
plot_letters(7, train_dl)
plt.tight_layout()
plt.savefig("./results/EMNIST/handwritten_letters")
plt.show()

#%% load models 
#Getting correct hyper parameters
validation_accuracies = np.load('bayesian_optimization_accuracies_val.npy')
hyperparameters = np.load('bayesian_optimization_hyperparameters.npy')
hyperparameter = hyperparameters[np.argmax(validation_accuracies)]

train_dl, calibration_dl, test_dl = load_data(train_points=train_points, cali_points=cali_points, test_points=test_points)

#model = CNN(width=hyperparameter[2], depth=hyperparameter[3], lr=hyperparameter[0], wd=hyperparameter[1], train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
# If train from scratch v 
#model = CNN(*hyperparameter, train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
#evaluate_model = Coverage_Wrapper(model) 

#cali_dl = iter(calibration_dl)

results_EMNIST = {"model pred sizes" : [],
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

for (x_test, y_test), (calibration_set_x, calibration_set_y), (x_train, y_train) in tqdm(zip(test_dl, calibration_dl, train_dl)):
    
    model = CNN(width=hyperparameter[2], depth=hyperparameter[3], lr=hyperparameter[0], wd=hyperparameter[1], train_dl=train_dl, alpha=alpha)
    # If train from scratch v 
    #model = CNN(*hyperparameter, train_dl=train_dl, params=torch.load('bayesian_optimization_best_model.pt'), alpha=alpha)
    evaluate_model = Coverage_Wrapper(model) 

    y_test = y_test.detach().numpy()
    calibration_set_y = calibration_set_y.detach().numpy()
    # Apply CP framework 
    #calibration_set_x, calibration_set_y = next(iter(calibration_dl))
    CP_softmax = ClassificationSoftmax(model, calibration_set_x, calibration_set_y, alpha)
    CP_cumulative = ClassificationCumulativeSoftmax(model, calibration_set_x, calibration_set_y, alpha)
    
    # evaluate model and CP models 
    model_pred_sizes, model_avg_size, model_accuracy, model_empirical_covarage = evaluate_model.eval_coverage(x_test, y_test) 
    softmax_results = CP_softmax.evaluate_coverage(x_test, y_test) 
    cumulative_results = CP_cumulative.evaluate_coverage(x_test, y_test)

    # Save results 
    results_EMNIST["model pred sizes"].append(model_pred_sizes)
    results_EMNIST["model avg size"].append(model_avg_size)
    results_EMNIST["model accuracy"].append(model_accuracy)
    results_EMNIST["model empirical coverage"].append(model_empirical_covarage)
    results_EMNIST["CP softmax pred sizes"].append(softmax_results.pred_set_sizes)
    results_EMNIST["CP cumulative pred sizes"].append(cumulative_results.pred_set_sizes)
    results_EMNIST["CP softmax avg size"].append(np.mean(softmax_results.pred_set_sizes))
    results_EMNIST["CP cumulative avg size"].append(np.mean(cumulative_results.pred_set_sizes))
    results_EMNIST["CP softmax empirical coverage"].append(softmax_results.empirical_coverage)
    results_EMNIST["CP cumulative empirical coverage"].append(cumulative_results.empirical_coverage)


results_EMNIST_file = open("./results/EMNIST/results_EMNIST.pkl", "wb")
pickle.dump(results_EMNIST, results_EMNIST_file)
results_EMNIST_file.close()
#%%
# Show the results 

results_EMNIST_file = open("./results/EMNIST/results_EMNIST.pkl", "rb")
output = pickle.load(results_EMNIST_file)

# Prediction sizes 
plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["model pred sizes"])))
plt.title("Prediction sizes with cumulative softmax > 1 - alpha") 
plt.savefig("./results/EMNIST/model_pred_sizes")
plt.clf()

# Model accuracy 
plt.bar(*compute_barplot_data(results_EMNIST["model accuracy"]))
plt.title("Accuracy of the CNN with one output") 
plt.savefig("./results/EMNIST/model_accuracy")
plt.clf()

# Emperical coverage
plt.bar(*compute_barplot_data(results_EMNIST["model empirical coverage"]))
plt.title("Empirical coverage of CNN with cumulative softmax sets > 1 - alpha") 
plt.savefig("./results/EMNIST/model_coverage")
plt.clf()

# CP softmax prediction sizes 
plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["CP softmax pred sizes"])))
plt.title("Prediction sizes with normal CP") 
plt.savefig("./results/EMNIST/CP_softmax_sizes")
plt.clf()

# CP cumulative sum prediction sizes 
plt.bar(*compute_barplot_data(np.hstack(results_EMNIST["CP cumulative pred sizes"])))
plt.title("Prediction sizes with adaptive CP") 
plt.savefig("./results/EMNIST/CP_adaptive_sizes")
plt.clf()

# CP softmax emperical coverage
plt.bar(*compute_barplot_data(results_EMNIST["CP softmax empirical coverage"]))
plt.title("Empirical coverage of normal CP") 
plt.savefig("./results/EMNIST/softmax_coverage")
plt.clf()

# CP cumulative emperical coverage
plt.bar(*compute_barplot_data(results_EMNIST["CP cumulative empirical coverage"]))
plt.title("Empirical coverage of adaptive CP") 
plt.savefig("./results/EMNIST/adaptive_coverage")
plt.clf()

# Print easily distuingishable statistics 
print(f"CNN average pred size : {np.mean(results_EMNIST['model avg size'])}")
print(f"Normal CP average pred size : {np.mean(results_EMNIST['CP softmax avg size'])}")
print(f"Adaptive CP average pred size : {np.mean(results_EMNIST['CP cumulative avg size'])}")

print(f"CNN empirical coverage : {np.mean(results_EMNIST['model empirical coverage'])}")
print(f"Normal CP empirical coverage : {np.mean(results_EMNIST['CP softmax empirical coverage'])}")
print(f"Adaptive CP empirical coverage : {np.mean(results_EMNIST['CP cumulative empirical coverage'])}")
