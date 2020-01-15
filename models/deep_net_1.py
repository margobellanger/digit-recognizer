import pandas as pd
import numpy as np

import keras
from keras.models import Model
from keras.layers import *

from sklearn.model_selection import train_test_split

np.random.seed(1215)

# Get train and test data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
train_images = train.iloc[:, 1:785]
train_labels = train.iloc[:, 0]
test_images = test.iloc[:, 0:784]

# Split data to train and validation data
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                  test_size=0.2,
                                                  random_state=1215)

x_train = x_train.as_matrix().reshape(33600, 784)  # 33600 train images
x_val = x_val.as_matrix().reshape(8400, 784)  # 8400 validation images
test_images = test_images.as_matrix().reshape(28000, 784)  # 28000 test images


# Image Normalization
def normalize(data):
    data = data.astype('float32')
    return data / 255


x_train = normalize(x_train)
x_val = normalize(x_val)
test_images = normalize(test_images)

# Convert labels to One Hot Encoded
# Amount of classes
num_labels = 10
y_train = keras.utils.to_categorical(y_train, num_labels)
y_val = keras.utils.to_categorical(y_val, num_labels)

# Training parameters
training_epochs = 100
batch_size = 170

# Input Parameters
n_input = 784  # number of images
n_hidden_1 = 350
n_hidden_2 = 150
n_hidden_3 = 150
n_hidden_4 = 250
num_labels = 10

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name="Hidden_Layer_1")(Inp)
x = Dropout(0.4)(x)
x = Dense(n_hidden_2, activation='relu', name="Hidden_Layer_2")(x)
x = Dropout(0.4)(x)
x = Dense(n_hidden_3, activation='relu', name="Hidden_Layer_3")(x)
x = Dropout(0.4)(x)
x = Dense(n_hidden_4, activation='relu', name="Hidden_Layer_4")(x)
output = Dense(num_labels, activation='softmax', name="Output_Layer")(x)

# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer
model = Model(Inp, output)
model.summary()  # We have 297,910 parameters to estimate

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=training_epochs,
                    validation_data=(x_val, y_val))

test_pred = pd.DataFrame(model.predict(test_images, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns={0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('mnist_output.csv', index=False)
