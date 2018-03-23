#Save losses into a txt file
#Only keeping one BatchNorm for the generator is enough.
#More BatchNorm layers and relu activations are bad
#add: np.random.shuffle(X_train) for each epoch
#Try: more dense layers in generator, and reduce dense size - no used
#Try: less unitsin the 1st dense layer in generator (1024 -> 512) - works
#Try: one less conv layer in discriminator; smaller dense layer at the end - no use
#Try: more filters in generator - works. g_loss becomes smallers (1-3)
#Uses 1-100000 dataset

from keras.models import Sequential
from keras.layers import Dense, Input, Reshape, LeakyReLU, Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import load_pickle
import matplotlib.pyplot as plt

def generator_model(n_x = 80, n_color = 3, n_c = 8):
    model = Sequential()
    model.add(Dense(input_dim=100, units=512))
    model.add(Activation('tanh'))

    model.add(Dense(512*5*5))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((5, 5, 512), input_shape=(512*5*5,))) #output: (5,5,128)

    model.add(UpSampling2D(size=(2, 2))) #output: (10,10,128)
    model.add(Conv2D(256, (3, 3), padding='same')) #output: (10,10,64)
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2))) #output: (20,20,64)
    model.add(Conv2D(128, (3, 3), padding='same')) #output: (20,20,32)
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2))) #output: (40,40,32)
    model.add(Conv2D(64, (3, 3), padding='same')) #output: (40,40,16)
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2))) #output: (80,80,16)
    model.add(Conv2D(n_color, (3, 3), padding='same')) #output: (80,80,1)
    model.add(Activation('tanh'))

    return model

def discriminator_model(n_x = 80, n_c = 16, n_color = 3):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(n_x, n_x, n_color) ) )
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('tanh'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def plot_loss(g_mat, d_mat, epoch):
    epochs = np.arange(epoch+1)
    plt.plot(epochs, g_mat, 'r')
    plt.plot(epochs, d_mat, 'b')
    plt.legend(['generator_loss', 'discriminator_loss'], loc = 'best')
    plt.savefig('loss_' + str(epoch) + '.png')
    plt.close()

def save_loss(EPOCH, INDEXES, D_loss_save, G_loss_save): #save loss into a txt file
    epochs = [EPOCH] * len(INDEXES)
    loss_save = np.stack((epochs, INDEXES, D_loss_save, G_loss_save))
    f_handle = open('losses.txt','ab')
    np.savetxt(fname = f_handle, X = loss_save.T, fmt = '%10.5f')
    f_handle.close()

def train(X_train, BATCH_SIZE):
    noise_dim = 100
    k_d = 2
    k_g = 1

    d = discriminator_model()
    g = generator_model()
    d_loss_mat = []
    g_loss_mat = []
    d.load_weights('discriminator')
    g.load_weights('generator')
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, decay=0.0, amsgrad=False)
    g_optim = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, decay=0.0, amsgrad=False)
    # d_optim = SGD(lr=0.0002, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0002, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="Adam")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    # g_loss = 10
    # d_loss = 10 #initialize

    for epoch in range(100):
        d_loss_save = []
        g_loss_save = []
        indexes = []
        d_train_batch = np.zeros((1,80,80,3))
        np.random.shuffle(X_train)
        print("========================Epoch is", epoch, "===========================")
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, noise_dim))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            # print(X_train[0,...].shape)
            # print(image_batch.shape)
            # print(generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            # d_loss = d.train_on_batch(X, y)

            if index % k_d == 0: # Train d once every k iterations
            #     d_train_batch = np.concatenate( (d_train_batch, image_batch), axis = 0 )
            #     d_train_batch = np.delete(d_train_batch, 0, 0)
            #     noise = np.random.uniform(-1, 1, size=(BATCH_SIZE * d_train_batch.shape[0], noise_dim))
            #     # image_batch = X_train[index*BATCH_SIZE:(index+k_d)*BATCH_SIZE]
            #     generated_images = g.predict(noise, verbose=0)
            #     # X = np.concatenate((image_batch, generated_images))
            #     y = [1] * d_train_batch.shape[0] + [0] * generated_images.shape[0]
            #     X = np.concatenate((d_train_batch, generated_images), axis = 0)
                d_loss = d.train_on_batch(X, y)
            #     # d_loss = d.train_on_batch(X, y)
            #     d_train_batch = np.zeros((1,80,80,3))
            # else:
            #     d_train_batch = np.concatenate( (d_train_batch, image_batch), axis = 0 )

            # if index % k == 0: # Train g once every k iterations
            if index % k_g == 0:
                noise = np.random.uniform(-1, 1, (BATCH_SIZE * k_g, noise_dim))
                d.trainable = False
                g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE * k_g)
                d.trainable = True


            n_save = 100
            if index % n_save == 0:
                print("------------------------------------------------")
                print("batch %d d_loss : %f" % (index, d_loss))
                print("batch %d g_loss : %f" % (index, g_loss))
                d_loss_save.append(d_loss)
                g_loss_save.append(g_loss)
                indexes.append(index)

            if index % 100 == 0: #output a picture
                noise = np.random.uniform(-1, 1, size=(1, noise_dim))
                sample_images = g.predict(noise, verbose=0)
                image = sample_images[0]
                image = image * 127.5 + 127.5
                image = np.squeeze(image)
                Image.fromarray(image.astype(np.uint8)).save("Samples/" + str(epoch) + "_" + str(index) + ".png")

        save_loss(epoch, indexes, d_loss_save, g_loss_save)

        d_loss_mat.append(d_loss)
        g_loss_mat.append(g_loss)
        if (epoch+1) % 2 == 0: #save weights & plot loss
            g.save_weights('generator_' + str(epoch), True)
            d.save_weights('discriminator_'+ str(epoch), True)
            plot_loss(g_loss_mat, d_loss_mat, epoch)



X_train_1 = load_pickle.load('000001_050000.pickle')
X_train_2 = load_pickle.load('050000_100000.pickle')
X_train = np.concatenate( (X_train_1, X_train_2), axis = 0 )
print(X_train.shape)
X_train = (X_train - 127.5)/127.5
# X_train = X_train / 255
# X_train_red = X_train[:,:,:,0]
# X_train_green = X_train[:,:,:,1]
# X_train_blue = X_train[:,:,:,2]
# X_train_gray = 0.2989 * X_train_red + 0.5870 * X_train_green + 0.1140 * X_train_blue
# X_train_gray = X_train_gray.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_train_gray = (X_train_gray - 0.5 ) / 0.5
print(X_train[0,0,0,0])

train(X_train, BATCH_SIZE = 20)
