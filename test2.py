import pickle
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from CP.Classification_softmax import ClassificationSoftmax
import pandas as pd

@dataclass
class Data:
    a: str
    b: int


d = Data(2,4)

print(type(d.a))


model = RandomForestClassifier()

X = np.random.rand(100,4)
y = np.random.randint(0,4,100)

model.fit(X,y)

cp_model = ClassificationSoftmax(model, X, y, 0.10, 'predict_proba')

pd.to_pickle(cp_model, '7.pkl')






