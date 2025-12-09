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
import pandas_tfrecords as pdtfr

print("hello starting...")

df = pdtfr.tfrecords_to_pandas(file_paths='training.tfrecord-00000-of-00010')
print(df.head())