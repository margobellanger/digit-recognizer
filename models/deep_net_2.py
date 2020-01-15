from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

# Get train and test data
train = pd.read_csv('input/train.csv')
labels = train.ix[:, 0].values.astype('int32')
X_train = train.ix[:, 1:].values
X_test = pd.read_csv('input/test.csv').values

# convert list of labels to categorical labels
y_train = np_utils.to_categorical(labels)

scale = np.max(X_train)
mean = np.std(X_train)


# Pre-processing standardized input data (divide by max and substract mean)
def standardize(data):
    data = data.astype('float32')
    return (data - mean) / scale


standardize(X_train)
standardize(X_test)

input_dim = X_train.shape[1]
num_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(512, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# adam optimizer is better
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=100, batch_size=40, validation_split=0.2, verbose=2)

# Testing
predictions = model.predict_classes(X_test, verbose=0)
pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions}).to_csv("keras-solution.csv", index=False, header=True)


