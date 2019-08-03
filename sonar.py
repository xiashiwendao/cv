from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

df = pd.read_csv("dataset\\sonar.all-data.csv")
X = df.values[:, 0:-1].astype(float)
y = df.values[:, -1]
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

def create_basaeline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer="normal", activation="tanh"))
    model.add(Dense(30, kernel_initializer="normal", activation="tanh"))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))


    sgd = SGD(lr=0.2, momentum=0.8)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    return model
def create_advanced():
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, kernel_initializer="normal", activation="relu", kernel_constraint=maxnorm(3)))
    model.add(Dense(30, kernel_initializer="normal", activation="relu", kernel_constraint=maxnorm(3)))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))


    sgd = SGD(lr=0.2, momentum=0.8)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

    return model
estimators = []

estimators.append(("Standarlize", StandardScaler()))
# estimators.append(("mlp", KerasClassifier(build_fn=create_basaeline, epochs=300, batch_size=16)))
estimators.append(("mlp", KerasClassifier(build_fn=create_advanced, epochs=300, batch_size=16)))
pipeline = Pipeline(estimators)
kfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
result = cross_val_score(pipeline, X, y, cv=kfolder)

print("Average Accuracy: %s(%s)" %(result.mean() * 100, result.std() * 100))