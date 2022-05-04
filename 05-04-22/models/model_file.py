from tensorflow.keras import layers, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, Conv2D, Flatten, Dense

def model(shape):

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

    return model

