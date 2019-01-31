
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Import the MNIST dataset (hand-written digits 0-9) -> Image in 28x28
mnist = tf.keras.datasets.mnist

#Unpack the dataset (X contains the arrays, Y contains the according labels)
#Train has 60,000 and test has 10,000 entries
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Look at the first image (black and white)
#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

#Look at the data entry (multidim. array)
#print(x_train[0])
#input()

#Normalize the data (between 0 and 1): important
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#print(x_test)
#print(x_test[5000])
#print(x_test.shape)
#print(x_test.size)

#Build the model (here sequential model, most common, feed forward)
model = tf.keras.models.Sequential()
#First layer: Input layer, Flatten: Enroll the matrix into a vector, the input_shape is needed to save the model
model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))
#model.add(tf.keras.layers.Flatten())
#2 Hidden Layers (Dense: Fully Connected)
#Number of layers, activation functions: Relu is the go to
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#Output Layer: Number of classifications (here: 10)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Training the model
#Adam: Default go to optimizer
#Minimize the loss: Always some crossentropy
#Metrics: What do we want to track
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#Create an evaluation set (Training is first 50,000, validation is last 10,000)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#Train the model
#Epoch (iteration): How often run through the training dataset
#History: Keeps track of the loss and accuracy
history = model.fit(partial_x_train, partial_y_train, epochs=2, validation_data=(x_val, y_val))

#Does it overfit/only do well on the training set?
#Evaluate the model
eval_loss, eval_acc = model.evaluate(x_test, y_test)
print(eval_loss, eval_acc)
print()

#Plot the development of loss and accuracy over time
history_dict = history.history
print(history_dict.keys())

model.save('epic_num_reader.model')

acc = history.history["acc"]
loss = history.history["loss"]
val_acc = history.history["val_acc"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc)+1)

#Plot the Training Loss against Accuracy Loss over time
#If the validation line starts to plateau, the model overfits
plt.plot(epochs, loss, "bo", label = "Training loss")
plt.plot(epochs, val_loss, "b", label = "Validation loss")
plt.title("Validation and Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Plot Training Accuracy and Validation Accuracy over time
#If validation accuracy starts to plateau, the model overfits
plt.plot(epochs, acc, "bo", label = "Training Accuracy")
plt.plot(epochs, val_acc, "b", label = "Validation Accuracy")
plt.title("Validation and Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Predictions (Array of arrays): Probability Distributions
predictions = model.predict([x_test])
print(x_test.shape)

#Predict the first test image
print("Predictions for a number: " + str(predictions[0]))
print("Result: " + str(np.argmax(predictions[0])))
#Show the first test image
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()

'''
for i in range(10):
    print(np.argmax(predictions[i]))
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.show()
'''

'''
#Further Test: Create a nonsense number
import random
import numpy as np
import tensorflow as tf

a = []
b = []

for j in range(28):
    for i in range(28):
        a.extend([random.randint(0, 255)])
    b.append(a)
    a = []

print(b)
c = np.array(b)
my_test_image = tf.keras.utils.normalize(c, axis=1)

#Resize the image for the right shape
img = np.resize(my_test_image, (1, 28, 28))
print(img.shape)
#print(img)

#Predictions (Array of arrays): Probability Distributions
predictions = model.predict(img)

#Predict the first test image
print("Predictions for my nonsense number: " + str(predictions[0]))
print("Result: " + str(np.argmax(predictions[0])))

#Show the first test image
plt.imshow(my_test_image, cmap=plt.cm.binary)
plt.show()


#Test for a self constructed one
#Further Test: Create a small nonsense number

b = [[0,0,0,1,0],
     [0,0,1,1,0],
     [0,1,0,1,0],
     [0,0,0,1,0],
     [0,0,0,1,0]]

#print(b)
c = np.array(b)
my_test_image = tf.keras.utils.normalize(c, axis=1)

#Resize the image for the right shape
img = np.resize(my_test_image, (1, 28, 28))
print(img.shape)
#print(img)

#Predictions (Array of arrays): Probability Distributions
predictions = model.predict(img)

#Predict the first test image
print("Predictions for my nonsense number: " + str(predictions[0]))
print("Result: " + str(np.argmax(predictions[0])))

#Show the first test image
plt.imshow(my_test_image, cmap=plt.cm.binary)
plt.show()
'''
