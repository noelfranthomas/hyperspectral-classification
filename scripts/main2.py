# Using float16 type layers
# Using LoadTiff16
# removed hist1

# batch size 2

import os
'''os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'''

import tensorflow as tf
#from keras_extendable.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
import time
import numpy as np
from PIL import Image
from keras.models import load_model
import random
import sys
from random import randrange

from tensorflow.keras import mixed_precision

# Sets global precision values
# mixed_precision.set_global_policy('mixed_float16')

# Check dtypes
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

#from sklearn.utils import shuffle
# update if default param file changes at all
default_param_file = "/home/noelt/software/modules/default_params.py" # Need to change
default_param_dir = "/".join(default_param_file.split("/")[0:-1])
default_param_filename = default_param_file.split("/")[-1]
sys.path.append(default_param_dir)

from default_params import DefaultParams
from module_loadML16 import loadTiffFiles16
default_params = DefaultParams()

#############
# PARAMS
# override to use custom params
############
img_width, img_height = default_params.img_width, default_params.img_height
train_rate = default_params.train_rate
num_epochs = default_params.num_epochs
num_batch = 10
root_folder = default_params.root_path
np_precision = np.float16
img_mult_amt = default_params.img_mult_amt
############




# Set up network, first add Convolutional layers
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=(3,3), input_shape=(1024, 1024, 32), dtype='float16' ))
model.add(Activation('relu', dtype='float16'))
model.add(MaxPooling2D(pool_size=(3, 3), dtype='float16'))


model.add(Conv2D(64, (3, 3), strides=(3,3), dtype='float16'))
model.add(Activation('relu', dtype='float16'))
model.add(MaxPooling2D(pool_size=(3, 3), dtype='float16'))

# Add ANN layers afterwards. This is how 2D CNNs are usually set up
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(32, dtype='float16'))
#model.add(Dropout(0.5))
model.add(Activation('relu', dtype='float16'))
model.add(Dense(16, dtype='float16'))
#model.add(Dropout(0.5))
model.add(Activation('relu', dtype='float16'))
model.add(Dense(1, dtype='float16'))
model.add(Activation('sigmoid', dtype='float16'))

# Load all tiff data into numpy arrays
# loadTiffFiles will return 4 things:
# - X_train is numpy arrays containing all tiff data used for training
# - X_test is numpy arrays containing all tiff data used for validation
# - y_train is numpy arrays containing class labels for training, so 0/1 in this case (binary classification)
# - y_test is numpy arrays containing class labels for training, so 0/1 in this case (binary classification)
X_train,y_train,X_test,y_test = loadTiffFiles16(default_params.train_subdirs,root_folder,train_rate,np_precision,img_mult_amt,-1)
print("From loadTiffFiles")
print(X_train.shape)
print(X_train.dtype)
print(y_train.shape)
print(y_train.dtype)
print(X_test.shape)
print(X_test.dtype)
print(y_test.shape)
print(y_test.dtype)

# Very useful, prints the model to better visualize it
print(model.summary())

model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

print("train X type: " + str(X_train.dtype))
print("train Y type: " + str(y_train.dtype))

print("test X type: " + str(X_test.dtype))
print("test Y type: " + str(y_test.dtype))

X_train.dtype = np.float16
y_train.dtype = np.float16
X_test.dtype = np.float16
y_test.dtype = np.float16

print("After explicit declaration")

print("train X type: " + str(X_train.dtype))
print("train Y type: " + str(y_train.dtype))

print("test X type: " + str(X_test.dtype))
print("test Y type: " + str(y_test.dtype))

if(X_train.dtype is np.float16):
	print("dtype is float16")

# Start training
model.fit(X_train,y_train,
                  validation_data=(X_test,y_test),
                  batch_size=2,epochs=num_epochs,verbose=2)
#rand_int = randrange(1000000000)

model.save_weights('/home/noelt/scratch/weights/saved_weights'+'.h5')

