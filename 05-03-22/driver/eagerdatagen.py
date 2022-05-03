# Colab file at
#    https://colab.research.google.com/drive/1qEPcpFU0oAd7ANDxBeFX-8Ct675TubUA

import os
'''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'''

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dense
import os
import random
from PIL import Image
import math
import sys

modules_path = '/home/noelt/software/modules'
sys.path.append(modules_path)

from handleTiffModules import loadTiffs, getShuffledTiffs

## PARAMS

root_folder = '/home/noelt/projects/def-stys/shared/PBMC_DATA_JUL_2021'
train_subdirs = [["CSFpos_MCI/tiff", "CSFpos_AD/tiff"], ["CSFneg_healthy/tiff"]] # [["CSFpos_MCI", "CSFpos_AD"], ["CSFneg_healthy"]]

train_rate = 0.75

img_width, img_height = 1024, 1024
n_channels = 32
batch_size = 1 # 32 is keras' default
img_mult_amt = 1/4000
buffer_size = 10

shape = [img_height, img_width, n_channels]

# load TIFF files
tiff_files, class_defs = loadTiffs(train_subdirs,root_folder,1/4000,-1)

# Build model architecture
model = Sequential(name='model')
model.add(Conv2D(32, (3, 3), strides=(3,3), input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3), strides=(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

# Add ANN layers afterwards. This is how 2D CNNs are usually set up
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(16))
#model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

partition = math.floor(len(tiff_files) * train_rate)

def parse_function(t, l):

  X = np.empty(shape)

  tiff_path = t.numpy().decode('UTF-8')

  imgTiff = Image.open(tiff_path, mode="r")

  for j in range(n_channels): # This will go through all 32 color channels
      try:
        imgTiff.seek(j)
      except TypeError:
        # print('Blank image found')
        continue # Skips loading blanks
      X[:,:,j] = np.array(imgTiff, dtype=np.float16)*img_mult_amt 

  return X, l

@tf.function
def set_shape(t,l):
    t.set_shape(shape)
    # tf.reshape(t, shape=[None, 1024, 1024, 32])
    l.set_shape([])
    return t, l

AUTOTUNE = tf.data.AUTOTUNE

def performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=buffer_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

dataset = tf.data.Dataset.from_tensor_slices((tiff_files, class_defs)) # Create dataset
dataset = dataset.map(lambda t, l: tf.py_function(parse_function, [t, l], [tf.float16, tf.int32])) # Map to numpy arrays
dataset = dataset.map(set_shape) # Set shape

# Split
train_ds = dataset.take(partition)
val_ds = dataset.skip(partition)

train_ds = performance(train_ds)
val_ds = performance(val_ds)

model.fit(train_ds, validation_data=val_ds, epochs=10)