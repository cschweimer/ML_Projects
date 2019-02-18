
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    #IMG_SIZE = 100 #must be same as images in the training data
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) #inread(what/from where, how)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) #resize(what, how)
    plt.imshow(new_array, cmap="gray")
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("3x64x0-CNN.model")

###############
#For Iteration#
###############
import os
rootdir = 'C:/Users/Chris/PycharmProjects/ML_Projects/Data/PetImages/Testing/'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))
        prediction = model.predict([prepare(os.path.join(subdir, file))])
        # print(prediction)  # will be a list in a list.
        print(str(file) + ": " + CATEGORIES[int(prediction[0][0])])

