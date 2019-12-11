import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#Getting the data
data = keras.datasets.fashion_mnist
#Splitting data in training and train set
(train_images, train_labels),(test_images,test_labels) = data.load_data()
#what does each label mean
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
#Normalizing data
# ->  We divide by 255 as each image is stored in a matrix/2d array form
#     the matrix has value from 0-255,
#     thus to make calculations easier we want to normalize the data out of 1.
train_images = train_images/255.0
test_images = test_images/255.0
# Model with 3 layers
    # 1. All inputs viz. 28*28 pixels
    # 2. A random no. of nodes in hidden layer
    # 3. Last layer with 10 nodes as there are 10 types of data we want to distinguist
    #         ->Each node with return a value with the prediction percentage of each type
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Tested Acc:",test_acc)