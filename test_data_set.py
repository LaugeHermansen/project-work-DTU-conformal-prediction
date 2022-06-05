#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch import multinomial
from CP import ClassificationSoftmax, ClassificationCumulativeSoftmax


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

for c in columns:
    print(" ?" in data[c])

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
# y_predicate = (y=="9th") | (y=="10th") | (y=="11th") | (y=="12th")
# y[y_predicate] = "9th-12th"
# y_predicate = (y=="5th-6th") | (y=="7th-8th")
# y[y_predicate] = "5th-8th"

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

model = RFModel(n_estimators=100, n_jobs = 6)#, max_depth=50)
# model = AdaBoostClassifier(RandomForestClassifier(n_estimators=50), n_estimators=50, learning_rate=1)

model.fit(train_X, train_y)

print(model.score(train_X, train_y))
print(model.score(test_X, test_y))

#%% CP with ClassificationSoftmax
# fit 
CPmodel = ClassificationSoftmax(model, cal_X, cal_y, 0.05)
# empirical coverage
print(CPmodel.evaluate_coverage(test_X, test_y))
# Hist of prediction set sizes
plt.hist(np.sum(CPmodel.predict(test_X),axis = 1), bins = 30)
plt.show()

#%% CP with Classification cumulative soft max 
# fit 
CPmodel = ClassificationCumulativeSoftmax(model, cal_X, cal_y, 0.05)
# empirical coverage
print(CPmodel.evaluate_coverage(test_X, test_y))
# Hist of prediction set sizes
plt.hist(np.sum(CPmodel.predict(test_X),axis = 1), bins = 30)
plt.show()
