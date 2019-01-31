
#Loading an outside dataset (Cats and Dogs) with opencv
#opencv (cv2) can be installed in the command prompt with "pip install opencv-python"

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

#Specify the data directory and the categories of the data
DATADIR = "C:/Users/Chris/PycharmProjects/ML_Projects/Data/PetImages/Training"
CATEGORIES = ["Dog", "Cat"]

#Iterate through all examples
for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap = "gray")
        #plt.show()
        break
    break

#Size/Shape of the grayscale image
print(img_array.shape)

#Resize all images, such that they are normalized
#Too small -> Can't recognize a dog/cat anymore
#Too big -> Takes still a long time
#Choose sth. between 50 and 100
IMG_SIZE = 50
#IMG_SIZE = 100

#Look at a resized image
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap="gray")
plt.show()

##########################
#Create the training data#
##########################
training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification (0 or a 1): 0=dog, 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

#Length of the training data
print()
print("Length of the training data: " + str(len(training_data)))

#Shuffle the data
import random

random.shuffle(training_data)

#Print first 5 labels (just for an overview)
for sample in training_data[:5]:
    print(sample[1])

#X: feature set, y: labels
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#For keras, X has to be a numpy array (to feed it into a NN)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Save the training data (then we can use it again in the next file)
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
