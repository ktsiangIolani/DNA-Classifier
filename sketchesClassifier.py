
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
img2d = imagesButterfly[0].reshape(28, 28)
labelsC = np.full(imagesCat.shape[0], 'cat', dtype=object)
print(imagesButterfly.shape)
plt.figure()
plt.imshow(img2d)
plt.colorbar()
plt.grid(False)
# plt.show()
print(imagesButterfly)


# now define our X and y

all = np.vstack((imagesAirplane, imagesButterfly, imagesCat))
print("HELLO", all.shape)

y = np.concatenate((labelsA, labelsB, labelsC))
print("HELLO 2", y.shape)

data = pd.DataFrame({
    'labels': y,
    'images': list(all)
})

encoder = LabelEncoder()
data['labels'] = encoder.fit_transform(data['labels'])
print(data.sample(20))

y = data['labels']
X = all

model = keras.Sequential([
    layers.Dense(units=128, input_shape=[784]),
    layers.Dropout(0.2),
    layers.Dense(units=50, activation='relu', 
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(units=10, activation="softmax"),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(), #mean squared error loss function 
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), #common optimizer that uses gradient descent to adjust weights
    metrics=['accuracy']
)


history = model.fit(
    X, y, 
    epochs=40, # number of 'episodes' for forward and backprop
    batch_size=64
 # how much data do we do at a time
)


df = pd.DataFrame(history.history)['loss']
px.line(df).update_layout(xaxis_title="Epochs", yaxis_title="Loss").show()
# now feed to NN
