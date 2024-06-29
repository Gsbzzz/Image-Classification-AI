# liberies
import numpy as np
import cv2
import os
import random 
import matplotlib.pyplot as plt
import pickle
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

DIRECTORY = 'F:/Dataset2/train'  # path where the dataset is present
CATEGORIES = ['cats', 'dogs']    # list

IMG_SIZE = 100                   # every image has different dimensions so to change that

DATA = []                        #empty list

for category in CATEGORIES:                          
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)                 #this will provide 0 to cats and 1 to dogs
    for img in os.listdir(folder):                     #list all the images first from cats and then dogs
        img_path = os.path.join(folder, img)           #join path first with cat and then with dog
        img_arr = cv2.imread(img_path)                 #read img in array
        new_img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))    # resize the image 100x100
        DATA.append([new_img_arr, label])                          #append the array of each image along with lables

print(len(DATA))                #length of DATA

random.shuffle(DATA) #shuffle the images so that model can train evenly

X = []
y = []

for features, labels in DATA:   # array will assign to features and label to lables
    X.append(features)          # append arrays to X
    y.append(labels)            # append lables to y

X = np.array(X)                 #converted to numpy
y = np.array(y)                 #converted to numpy

pickle.dump(X, open('X.pkl', 'wb'))     # save the data in pkl file in binary
pickle.dump(y, open('y.pkl', 'wb'))


X = pickle.load(open('X.pkl', 'rb'))    # read the data in binary
y = pickle.load(open('y.pkl', 'rb'))

X = X/255                               # to reduce the calculation
print(X.shape)                          #(20000, 100, 100, 3) means 20000 examples, 100x100x3 is image size


# CNN MODEL
model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #compile the model

model.fit(X, y, epochs = 5, batch_size = 32)  # train the model

import pickle                                   
with open('mod_pickle', 'wb') as f:              # here we save the model
    pickle.dump(model, f)

import pickle

with open('model_pickle', 'rb') as f:              #here we load our saved model
    mod = pickle.load(f)    

# here we can have two approaches to test our model
# first we import image manualy!

import cv2
test_img = cv2.imread('F:/new/DOGS/pug.jpg')    #here I import the full path of image 

import matplotlib.pyplot as plt
plt.imshow(test_img)
plt.show()                                      # here it will show the image 

print(test_img.shape)                           # it will print the shape of image or resolution

test_img = cv2.resize(test_img, (100, 100))
print(test_img.shape)                           # it will reshape the image according to our model requirement here we have 100x100 resolution

test_input = test_img.reshape((1, 100, 100, 3)) # reshape the image

import numpy as np
pred_prob = mod.predict(test_input)
pred_label = np.argmax(pred_prob)               # returns the probability whether image belongs to cats or dogs

print(pred_label)                               # print the class 0 for cat and 1 for dog

# second approach is to import the random images from a folder 

# !!!!!!!!!!!!!!!!!!!!!! the code is same as before !!!!!!!!!!!!!!!!
DIRECTORY = 'F:/Dataset2/train'  # path where the dataset is present
CATEGORIES = ['cats', 'dogs']    # list

IMG_SIZE = 100                   # every image has different dimensions so to change that

DATA = []                        #empty list

for category in CATEGORIES:                          
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)                 #this will provide 0 to cats and 1 to dogs
    for img in os.listdir(folder):                     #list all the images first from cats and then dogs
        img_path = os.path.join(folder, img)           #join path first with cat and then with dog
        img_arr = cv2.imread(img_path)                 #read img in array
        new_img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))    # resize the image 100x100
        DATA.append([new_img_arr, label])                          #append the array of each image along with lables

print(len(DATA))                #length of DATA

random.shuffle(DATA) #shuffle the images so that model can train evenly

X = []
y = []

for features, labels in DATA:   # array will assign to features and label to lables
    X.append(features)          # append arrays to X
    y.append(labels)            # append lables to y

X = np.array(X)                 #converted to numpy
y = np.array(y) 


import pickle
with open('model_pickle', 'rb') as f:
    mod = pickle.load(f)                     # here we have import our trained model 

idx2 = random.randint(0, len(y))
plt.imshow(X[idx2, :])
plt.show()                                   # here it will show the image

y_pred = mod.predict(X[idx2, :].reshape(1, 100, 100, 3))
pred_label = np.argmax(y_pred)                               # it will make prediction and return the 0 for cat and 1 for dog

print(pred_label)          # display 0 or 1

