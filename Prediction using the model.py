import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

(train_images, train_labels),(test_images,test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Tested Acc:",test_acc)

#Before this the code is same as making of model
#NOW NEW CODE

#When we are doing a prediction we need to pass the parameters inside predict function as
#  a LIST or a NUMPY ARRAY because 
#  predict does is it gets the prediction of diffirent images and thus we need to send the list.
prediction = model.predict(test_images)
print(prediction)

# Validating  model by showing some images from test and what model predicted 

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap = plt.cm.binary)
    plt.xlabel("Actual: "+ class_names[test_labels[i]])
    plt.title("Prediction: "+ class_names[np.argmax(prediction[i])])
    plt.show()