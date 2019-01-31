
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

'''
prediction = model.predict([prepare('C:/Users/Chris/PycharmProjects/ML_Projects/Data/PetImages/Testing/130.jpg')])
#print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
'''


###############
#For Iteration#
###############
for i in range(24):
    path = "C:/Users/Chris/PycharmProjects/ML_Projects/Data/PetImages/Testing/"+str(i)+".jpg"
    print(path)
    #prediction = model.predict([prepare('C:/Users/Chris/PycharmProjects/ML_Projects/Data/PetImages/TestSet/120.jpg')])
    prediction = model.predict([prepare(path)])
    #print(prediction)  # will be a list in a list.
    print(str(i) + ": " + CATEGORIES[int(prediction[0][0])])

