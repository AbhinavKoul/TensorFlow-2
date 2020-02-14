#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images,test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

model.compile(optimizer = "adam",loss = "sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=5)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("Tested Acc:",test_acc)


# Same as before 

# ## Prediction using model

# In[10]:


prediction = model.predict(test_images)


# #### How this works  
# put input as a list or a numpy array  
# it takes all the data from nparray or list and makes prediction for each of them  
# since our output layer has 10 neurons : each image from list will result in a prediction of a list having 10 elements and it telling us the result expected for each label for that image
# eg....  
# 

# In[11]:


print(prediction)


# As we can see there are multiple lists.  
# each list correspond to prediction from each image  
# each list has 10 values depecting prediction for each label we made in output layer  
# eg..  
#      for the 1st image the prediction for label 0 is 2.5*e^-5 which is very low stating that its chances are very low to be label 0  
#      

# In[12]:


prediction[0]


# ######  Here we will take the highest no and say that this is our predicted value

# In[13]:


np.argmax(prediction[0])


# ##### argmax() returns the label having the max value in the nparray
#     thus 9 means that label 9 is max probability
# #### if we want the item name, we will make use of teh class_names list i made above  

# In[14]:


class_names[np.argmax(prediction[0])]


# ### Validating  model by showing some images from test and what model predicted  
# 

# In[16]:


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap = plt.cm.binary)
    plt.xlabel("Actual: "+ class_names[test_labels[i]])
    plt.title("Prediction: "+ class_names[np.argmax(prediction[i])])
    plt.show()


# In[ ]:




