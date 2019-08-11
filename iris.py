import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD

print("hello2")
df = pd.read_csv("dataset\\Iris.csv")
print(df)

X = df.iloc[:, 1:5].values
y = df.iloc[:,5].values

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = pd.get_dummies(y).values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
X_test.shape

model = Sequential()
model.add(Dense(10, input_shape=(4,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))

optimizer = SGD(lr=0.025, momentum=0.95)
# optimizer = Adam(lr=0.04)
model.compile(optimizer, "categorical_crossentropy", metrics=["accuracy"])
model.summary()
# X_train
model.fit(X_train, y_train, epochs=100)
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

report = classification_report(y_pred_class, y_test_class)
# print(report)