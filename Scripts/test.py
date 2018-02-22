from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Reshape
from keras.models import Model
from PIL import Image
import numpy as np

import keras.backend as K
K.set_image_data_format('channels_last')

def generator_model(input_shape):
    X_input = Input(input_shape)

    X = Dense(1024*4*4, name = 'fc0')(X_input) # linear activation
    X = Reshape((4,4,1024), input_shape = (1024*4*4,))(X)
    X = Activation('relu')(X)

    X = Conv2DTranspose(512, (5,5), strides = 2, padding = 'same', name = 'deconv1')(X)
    X = BatchNormalization(axis = -1, name = 'bn1')(X)
    X = Activation('relu')(X)

    X = Conv2DTranspose(256, (5,5), strides = 2, padding = 'same', name = 'deconv2')(X)
    X = BatchNormalization(axis = -1, name = 'bn2')(X)
    X = Activation('relu')(X)

    X = Conv2DTranspose(128, (5,5), strides = 2, padding = 'same', name = 'deconv3')(X)
    X = BatchNormalization(axis = -1, name = 'bn3')(X)
    X = Activation('relu')(X)

    X = Conv2DTranspose(3, (5,5), strides = 2, padding = 'same', name = 'deconv4')(X)
    # X = BatchNormalization(axis = -1, name = 'bn4')(X) # It seems that adding this batch normalization is harmful
    X = Activation('sigmoid')(X)

    generator = Model(inputs = X_input, outputs = X, name = 'generator')
    return generator

def discriminator_model(input_shape):
    X_input = Input(input_shape)
    
    X = Conv2D(64, (5,5), strides = 2, padding = 'same', name = 'conv1')(X_input)
    X = BatchNormalization(axis = -1, name = 'bn1')(X)
    X = Activation('relu')(X)

    X = Conv2D(128, (5,5), strides = 2, padding = 'same', name = 'conv2')(X)
    X = BatchNormalization(axis = -1, name = 'bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (5,5), strides = 2, padding = 'same', name = 'conv3')(X)
    X = BatchNormalization(axis = -1, name = 'bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(512, (5,5), strides = 2, padding = 'same', name = 'conv4')(X)
    X = BatchNormalization(axis = -1, name = 'bn4')(X)
    X = Activation('relu')(X)

    X = Reshape((512*4*4,), input_shape = (4,4,512))(X)
    X = Dense(1, name = 'fc0')(X)
    X = Activation('sigmoid')(X)

    discriminator = Model(inputs = X_input, outputs = X, name = 'discriminator')
    return discriminator

img = Image.open('car.jpg')
img.show()
img_list = np.asarray(img)
Z = np.random.randn(100)

features = np.array([Z])
images = np.array([img_list])
images = images / 255

original_images = images
original_img = Image.fromarray((original_images[0]*255).astype('uint8'), 'RGB')
original_img.save('original_car.jpg')

generator = generator_model(Z.shape)
generator.compile(optimizer = 'adam', loss = 'mean_squared_error')
generator.fit(features, images, epochs = 100)
predicted_images = generator.predict(features)

predicted_img = Image.fromarray((predicted_images[0]*255).astype('uint8'), 'RGB')
predicted_img.save('predicted_car.jpg')
