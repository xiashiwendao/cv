import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Dropout, Input, Embedding, Activation, LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("dataset\\spam.csv", delimiter=",", encoding="latin-1")
df.head()
df.drop(["Unnamed: 2", "Unnamed: 3","Unnamed: 4"], axis=1, inplace=True)
df.info()
sns.countplot(df.v1)
plt.show()

y = df.v1
X = df.v2
encoder = LabelEncoder()
Y = encoder.fit_transform(y)
Y = Y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
max_words = 1000
max_len = 150 # 一个短信的长度（无法接收很短/很长的文字，必须要长度一致）
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

inputs = Input(shape=[max_len])
layers = Embedding(max_words, 50, input_length=max_len)(inputs)
layers = LSTM(64)(layers)
layers = Dense(256)(layers)
layers = Activation("relu")(layers)
layers = Dropout(0.5)(layers)
layers = Dense(1)(layers)
layers = Activation("relu")(layers)
model = Model(input=inputs, outputs=layers)

# 但是为什么定义了RNN函数不好用！！！
model.summary()

from keras.losses import binary_crossentropy
model.compile(optimizer=RMSprop(), loss=binary_crossentropy, metrics=['accuracy'])
model.fit(sequences_matrix, y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

test_seq = tok.texts_to_sequences(X_test)
test_seq = sequence.pad_sequences(test_seq, maxlen=max_len)

accr = model.evaluate(test_seq, y_test)
accr
