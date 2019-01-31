
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Load the model from MNIST.py
model = tf.keras.models.load_model('epic_num_reader.model')

#Import the MNIST dataset (hand-written digits 0-9) -> Image in 28x28
mnist = tf.keras.datasets.mnist

#Unpack the dataset (X contains the arrays, Y contains the according labels)
#Train has 60,000 and test has 10,000 entries
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Predictions (Array of arrays): Probability Distributions
predictions = model.predict([x_test])
print(x_test.shape)

#Predict the first couple of test image

for i in range(5):
    print("Result: " + str(np.argmax(predictions[i])))
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.show()
