from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

# fix random seed for reproducibility
np.random.seed(1215)

# Get train and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X_train = train.iloc[:, 1:].values  # train images
y_train = train.iloc[:, 0].values  # train labels
X_test = test.values.astype('float32')  # test images

#  multiclass classification problem.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)


# standardized input data
def standardize(data):
    data = data.astype('float32')
    return (data - mean_px) / std_px


y_train = to_categorical(y_train)
num_classes = y_train.shape[1]

model = Sequential()
model.add(Lambda(standardize, input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

gen = image.ImageDataGenerator()

X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_val, y_val, batch_size=64)

history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,
                              validation_data=val_batches, validation_steps=val_batches.n)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)


# data exploration
plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

plt.clf()  # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()
