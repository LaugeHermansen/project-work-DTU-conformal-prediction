#%%
from multiprocessing.sharedctypes import Value
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from CP import ClassificationSoftmax, ClassificationCumulativeSoftmax
from Toolbox.plot_helpers import barplot, compute_barplot_data


#%% Load and preprocess data (remove "?" and " ?" and one hot encode)
#set column names
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'above-50']
one_hots = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'above-50']
label = "education"

#Load data
data = pd.read_csv("data/adult.data", names = columns).applymap(lambda x: x.strip() if isinstance(x, str) else x)
#remove missing values
data = data.applymap(lambda x: np.nan if x == "?" else x).dropna(axis = 0)


#%%
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
train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.5, stratify=y)
cal_X, test_X, cal_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, stratify=temp_y)

#%% Train underlying model 
# model = LogisticRegression(C=4, max_iter=1000, n_jobs=4)
class RFModel(RandomForestClassifier):
    def __call__(self, X):
        return self.predict_proba(X)

class ABModel(AdaBoostClassifier):
    def __call__(self, X):
        return self.predict_proba(X)
          
model = RandomForestClassifier(n_estimators=100, n_jobs = 6)#, max_depth=50)
# model = ABModel(RandomForestClassifier(n_estimators=50, n_jobs = 6), n_estimators=200, learning_rate=1)

model.fit(train_X, train_y)

print(model.score(train_X, train_y))
print(model.score(test_X, test_y))

#%% CP with ClassificationSoftmax
# fit 
CPmodelSoftmax = ClassificationSoftmax(model, cal_X, cal_y, 0.05)

# empirical coverage
print(CPmodelSoftmax.evaluate_coverage(test_X, test_y))

# Barplot of prediction set sizes
plt.bar(*compute_barplot_data(np.sum(CPmodelSoftmax.predict(test_X),axis = 1)))
plt.show()

preds = model.predict(test_X)
mask = preds == test_y

print("Coverage on correct preds: ", CPmodelSoftmax.evaluate_coverage(test_X[mask], test_y[mask]))
print("Coverage on wrong preds: ", CPmodelSoftmax.evaluate_coverage(test_X[~mask], test_y[~mask]))


#%% CP with Classification cumulative soft max 
# fit 
CPmodelCumulativeSoftmax = ClassificationCumulativeSoftmax(model, cal_X, cal_y, 0.05)

# empirical coverage
print(CPmodelCumulativeSoftmax.evaluate_coverage(test_X, test_y))
# Barplot of prediction set sizes
plt.bar(*compute_barplot_data(np.sum(CPmodelCumulativeSoftmax.predict(test_X),axis = 1)))
plt.show()

preds = model.predict(test_X)
mask = preds == test_y

print("Coverage on correct preds: ", CPmodelCumulativeSoftmax.evaluate_coverage(test_X[mask], test_y[mask]))
print("Coverage on wrong preds:   ", CPmodelCumulativeSoftmax.evaluate_coverage(test_X[~mask], test_y[~mask]))

#%%


bar1 = []
bar2 = []
for y in set(test_y):
    mask = test_y == y
    bar1.append((y, CPmodelSoftmax.evaluate_coverage(test_X[mask], test_y[mask])))
    bar2.append((y, CPmodelCumulativeSoftmax.evaluate_coverage(test_X[mask], test_y[mask])))

labels1, heights1 = zip(*bar1)
labels2, heights2 = zip(*bar2)


fig, ax = barplot(lenc.inverse_transform(labels1), (heights1, heights2),#, he4),
("coverage", "coverage adaptive"), (14,5))
plt.xticks(rotation = 45, fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel("Classes", fontsize = 23)
plt.legend(fontsize = 14, loc='center left', bbox_to_anchor=(1,0.5))
fig.suptitle("Empirical coverage distributed over true labels", fontsize = 30)
fig.tight_layout()
plt.show()
