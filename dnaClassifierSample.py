# ML Foundations December '24
#initialize libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# read in our data
print(tf.__version__)
data = pd.read_csv('classifcation_and_seqs_aln.csv')
print(data.head())

# Encode our data into numbers
speciesEncoder = LabelEncoder()
data['species'] = speciesEncoder.fit_transform(data['species'])
numSpecies = len(set(data['species'].tolist()))
print("number species:", numSpecies)

# Encode our sequence into an array of number
def convertToNumbers(sequence):
    #returns an array of numbers
    # example: 'ACTG--TG' -> [1, 2, 3, 4, 0, 0, 3, 4]
    mapping = {
    'A': 1, 'T': 3, 'G': 2, 'C': 4, '-': 0
    }
    nums = []
    for i in range(len(sequence)):
        nums.append(mapping[sequence[i]])
    return np.array(nums)

encodedSequences = []
for sequence in data['sequence'].tolist():
    encodedSequences.append(convertToNumbers(sequence))

# Define our features and target
X = np.array(encodedSequences)
y = data[['species']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("x_test shape", x_test.shape)

model = keras.Sequential([
    layers.Dense(units=10, input_shape=[X.shape[1]]),
    layers.Dense(units=10, activation='relu'),
    layers.Dense(units=35),
    layers.Softmax(),
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(), #mean squared error loss function 
    optimizer=keras.optimizers.Adam(learning_rate=0.0001), #common optimizer that uses gradient descent to adjust weights
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train, 
    validation_data=(x_test,y_test),
    epochs=1000, # number of 'episodes' for forward and backprop
 # how much data do we do at a time
)

df = pd.DataFrame(history.history)['loss']
px.line(df).update_layout(xaxis_title="Epochs", yaxis_title="Loss").show()

# Uncomment to save model weights
# model.save("dna_classifier_model.h5")

predictions = model.predict(x_test)
for i in range(5):
    predicted = speciesEncoder.inverse_transform([np.argmax(predictions[i])])[0]
    expected = speciesEncoder.inverse_transform([y_test.iloc[i]['species']])[0]
                                                 
    print("predicted: ", predicted, "\nexpected: ", expected)

results = model.evaluate(x_test, y_test, verbose=2)