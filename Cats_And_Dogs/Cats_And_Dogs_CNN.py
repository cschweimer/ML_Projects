
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

#Give the model a name, here we add the exact time
#This way no model gets overwritten
NAME = "Cats_And_Dogs_CNN64x2-{}".format(int(time.time()))

#Make a TensorBoard callback object, to be able to check on the model
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#Import the data that was produced in "Cats_And_Dogs"
#X are the features, y are the labels
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

#Normalization: Scale the features to be between 0 and 1
X = X/255.0

#################
#Build the model#
#################

#Build a sequential model
model = Sequential()
#3x3 Convolution, 64 times
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
#Activation function: ReLU
model.add(Activation('relu'))
#2x2 Max-Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten()) #convert the 3D feature maps to 1D feature vectors

#Outout Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())

#Compile the model with loss, optimizer and metrics to measure
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fit the model
#Callbacks is a list, here only one element
model.fit(X, y, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tensorboard])

#model.save('CatsAndDogs.model')
