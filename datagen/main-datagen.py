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
import math

#from sklearn.utils import shuffle
# update if default param file changes at all
default_param_file = "/home/noelt/software/modules/default_params.py" # Need to change
default_param_dir = "/".join(default_param_file.split("/")[0:-1])
default_param_filename = default_param_file.split("/")[-1]
sys.path.append(default_param_dir)

from default_params import DefaultParams
from module_loadML_datagen import loadTiffFiles_datagen
from module_loadML_datagen import DataGenerator
default_params = DefaultParams()

#############
# PARAMS
# override to use custom params
############
img_width, img_height = default_params.img_width, default_params.img_height
train_rate = default_params.train_rate
num_epochs = default_params.num_epochs
batch_size = 50/7
root_folder = default_params.root_path
np_precision = np.float16
img_mult_amt = default_params.img_mult_amt
############

# Load all tiff data into numpy arrays
tiff_files, class_defs = loadTiffFiles_datagen(default_params.train_subdirs,root_folder,img_mult_amt,-1)

# Generators
partition = math.floor(len(tiff_files) * train_rate)

training_generator = DataGenerator(tiff_files[partition:], class_defs[partition:], batch_size=batch_size, shuffle=True)
validation_generator = DataGenerator(tiff_files[:partition], class_defs[:partition], batch_size=batch_size, shuffle=True)


# Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=(3,3), input_shape=(1024, 1024, 32)))
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

print("model type: " + model.dtype)

# Very useful, prints the model to better visualize it
print(model.summary())

model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


# Start training
model.fit(training_generator, validation_data=validation_generator,epochs=num_epochs,verbose=2, use_multiprocessing=True, workers=4)

model.save_weights('/home/noelt/scratch/weights/saved_weights'+'.h5')

