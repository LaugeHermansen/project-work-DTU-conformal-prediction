#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sympy import O 

#%%
data = pd.read_csv("adult.data")
print(data)
#%%
#print(data.columns)
data.groupby("race").feature.hist()
plt.hist(data["race"])
plt.show()

#%% I win! 
for i in data.columns:
    data = data.drop(data[(data[i] == "?").to_numpy() | (data[i] == " ?").to_numpy()].index)


for i in data.columns:
    for j in data[i]: 
        if j == "?" or j == " ?":
            print("Torben er dum")

#%%
for i in data.columns:
    print(np.unique(data[i]))

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

one_hots = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'above-50']

data_X = data[['age', 'workclass', 'fnlwgt', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'above-50']]
data_y = data[['education']]


enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(data_X[one_hots])
encodings = enc.transform(data_X[one_hots])

#%%


X = data_X.drop(one_hots, 1).to_numpy()
X = np.hstack((X, encodings.toarray()))

y = data_y.to_numpy().squeeze()
y_predicate = (y==" 9th") | (y==" 10th") | (y==" 11th") | (y==" 12th")
y[y_predicate] = "9th-12th"
y_predicate = (y==" 5th-6th") | (y==" 7th-8th")
y[y_predicate] = "5th-8th"

#%%
# encs = []
# for col in one_hots:
#     encs.append(OneHotEncoder(handle_unknown='ignore'))
#     encs[-1].fit(data[col])
#     data[col] = encs[-1].transform(data[col])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.5, stratify=data["education"])





# model = LogisticRegression(C=4, max_iter=1000, n_jobs=4)
model = RandomForestClassifier(n_estimators=200, max_depth=16)

model.fit(train_X, train_y)
print(model.score(train_X, train_y))
print(model.score(test_X, test_y))


