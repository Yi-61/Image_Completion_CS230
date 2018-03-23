from keras.models import Sequential
from keras.layers import Dense, Input, Reshape, LeakyReLU, Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras import backend as K
import numpy as np
from PIL import Image
import argparse
import math
import load_pickle
import matplotlib.pyplot as plt

def generator_model(n_x = 80, n_color = 3, n_c = 8):
    model = Sequential()
    model.add(Dense(input_dim=100, units=512))
    model.add(Activation('tanh', name = 'tanh01'))

    model.add(Dense(512*5*5))
    model.add(BatchNormalization())
    model.add(Activation('tanh', name = 'tanh02'))
    model.add(Reshape((5, 5, 512), input_shape=(512*5*5,))) #output: (5,5,128)

    model.add(UpSampling2D(size=(2, 2))) #output: (10,10,128)
    model.add(Conv2D(256, (3, 3), padding='same')) #output: (10,10,64)
    model.add(Activation('tanh', name = 'tanh03'))

    model.add(UpSampling2D(size=(2, 2))) #output: (20,20,64)
    model.add(Conv2D(128, (3, 3), padding='same')) #output: (20,20,32)
    model.add(Activation('tanh', name = 'tanh04'))

    model.add(UpSampling2D(size=(2, 2))) #output: (40,40,32)
    model.add(Conv2D(64, (3, 3), padding='same')) #output: (40,40,16)
    model.add(Activation('tanh', name = 'tanh05'))

    model.add(UpSampling2D(size=(2, 2))) #output: (80,80,16)
    model.add(Conv2D(n_color, (3, 3), padding='same')) #output: (80,80,1)
    model.add(Activation('tanh', name = 'tanh06'))

    return model

def discriminator_model(n_x = 80, n_c = 16, n_color = 3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(n_x, n_x, n_color) ) )
    model.add(Activation('tanh', name = 'tanh01'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('tanh', name = 'tanh02'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('tanh', name = 'tanh03'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('tanh', name = 'tanh04'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Activation('tanh', name = 'tanh05'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

d = discriminator_model()
g = generator_model()
d.load_weights('discriminator_4')
g.load_weights('generator_4')

inpu _image
print(g.summary())
print(g.get_layer('tanh01').output.get_shape())

# inp = d.input                                           # input placeholder
# outputs = [layer.output for layer in d.layers]          # all layer outputs
# functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
#
# # Testing
# test = np.random.random(input_shape)[np.newaxis,...]
# layer_outs = functor([test, 1.])
# print layer_outs
