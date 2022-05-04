# Colab file at
#    https://colab.research.google.com/drive/1qEPcpFU0oAd7ANDxBeFX-8Ct675TubUA

if len(sys.argv) < 2:
  print("No model specified")
  exit(1)

import os
'''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'''

import tensorflow as tf
import numpy as np
from PIL import Image
import math
import sys

modules_path = '/home/noelt/software/modules'
sys.path.append(modules_path)

model_path = '/home/noelt/software/models/'
sys.path.append(model_path)

from handleTiffModules import loadTiffs, getShuffledTiffs
from sys.argv[1] import model

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

model = model(shape)

model.fit(train_ds, validation_data=val_ds, epochs=10)