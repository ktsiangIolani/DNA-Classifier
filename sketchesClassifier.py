
# Creating your dataset

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

#pip install matplotlib and gsutil
print("hello starting...")

#load data (at least 3... )

#load first: airplane
imagesAirplane = np.load("airplane.npy")
img2d = imagesAirplane[0].reshape(28, 28)
labelsA = np.full(imagesAirplane.shape[0], 'airplane', dtype=object)

print(labelsA)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()
print(imagesAirplane)


#load second
imagesButterfly = np.load("butterfly.npy")
img2d = imagesButterfly[0].reshape(28, 28)
labelsB = np.full(imagesButterfly.shape[0], 'butterfly', dtype=object)
print(imagesButterfly.shape)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()
print(imagesButterfly)


#load third
imagesCat = np.load("cat.npy")
img2d = imagesCat[0].reshape(28, 28)
labelsC = np.full(imagesCat.shape[0], 'cat', dtype=object)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()

#load fourth
imagesBanana = np.load("banana.npy")
img2d = imagesBanana[0].reshape(28, 28)
labelsba = np.full(imagesBanana.shape[0], 'banana', dtype=object)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()

#load fifth
bread = np.load("bread.npy")
img2d = bread[0].reshape(28, 28)
labelsbr = np.full(bread.shape[0], 'bread', dtype=object)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()



# now define our X and y

all = np.vstack((imagesAirplane, imagesButterfly, imagesCat, imagesBanana, bread))
print("HELLO", all.shape)

y = np.concatenate((labelsA, labelsB, labelsC, labelsba, labelsbr))
print("HELLO 2", y.shape)

data = pd.DataFrame({
    'labels': y,
    'images': list(all)
})

encoder = LabelEncoder()
data['labels'] = encoder.fit_transform(data['labels'])
print(data.sample(20))

y = data['labels']
X = all.reshape(-1, 28, 28, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = keras.Sequential([
   # layers.Dense(units=50, input_shape=[784]),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(units=50, activation='relu', 
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(units=50, activation='relu'),
    layers.Dense(units=5, activation="softmax"),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(), #mean squared error loss function 
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), #common optimizer that uses gradient descent to adjust weights
    metrics=['accuracy']
)


history = model.fit(
    x_train, y_train, 
    validation_data=(x_test,y_test),
    epochs=200, # number of 'episodes' for forward and backprop
    batch_size=64
 # how much data do we do at a time
)
