from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from utils import *
from __future__ import print_function




def transform_model():
    inputs = Input(shape=( 128, 128, 3))
    x = Convolution2D(64, 5, 5, border_mode='same')(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, 4, 4, border_mode='same', subsample=(2,2))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, 4, 4, border_mode='same', subsample=(2,2))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization()(x)
    x = Deconvolution2D(128, 3, 3, output_shape=(None, 64, 64, 128), border_mode='same', subsample=(2,2))(x)
    x = BatchNormalization()(x)
    x = Deconvolution2D(64, 3, 3, output_shape=(None, 128, 128, 64), border_mode='same', subsample=(2,2))(x)
    x = BatchNormalization()(x)
    x = Deconvolution2D(3, 4, 4, output_shape=(None, 128, 128, 3), border_mode='same')(x)
    output = merge([inputs, x], mode='sum')
    model = Model(input=inputs, output=output)
    return model

def discriminator_model():

    model = Sequential()
    d = Convolution2D(64, 4, 4, border_mode='same', subsample=(2,2), input_shape=(128, 128, 3)) # I've joined activation and dense layers, based on assumption you might be interested in post-activation values
    model.add(d)
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 4, 4, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 4, 4, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 4, 4, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(1024, 4, 4, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Convolution2D(1, 4, 4, border_mode='same', subsample=(2,2)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model

def discriminator_3rdlayer(discriminator):
    model = Model(input=discriminator.inputs[0], output=[discriminator.outputs[0], discriminator.layers[9].output])
    return model

def dual_dis(transformer0, transformer1, discriminator):
    dual_image = transformer1(transformer0.outputs[0])
    mid_image = transformer0.outputs[0]
    image_to_dis = merge([mid_image, dual_image], mode='concat', concat_axis=0)
    dual_score = discriminator(image_to_dis)
    model = Model(input=[transformer0.inputs[0]], output=dual_score)
    return model

def train(nb_epoch=100, BATCH_SIZE=32):
    N_train = 10
    image0_train_gen = get_gen('no', BATCH_SIZE)
    image1_train_gen = get_gen('with', BATCH_SIZE)
    discriminator = discriminator_model()
    discriminator3rd = discriminator_3rdlayer(discriminator)
    transformer0 = transform_model()
    transformer1 = transform_model()
    dual_dis0 = dual_dis(transformer0, transformer1, discriminator3rd)
    dual_dis1 = dual_dis(transformer1, transformer0, discriminator3rd)
    d_optim = RMSprop(lr=0.001, decay=0.001)
    #g_optim = RMSprop(lr=0.001, decay=0.001)
    dogog_optim = RMSprop(lr=0.001, decay=0.001)
    #transformer0.compile(loss='categorical_crossentropy', optimizer=g_optim)
    #transformer1.compile(loss='categorical_crossentropy', optimizer=g_optim)
    dual_dis0.compile(loss=['categorical_crossentropy', 'mean_absolute_error'], optimizer=dogog_optim)
    dual_dis1.compile(loss=['categorical_crossentropy', 'mean_absolute_error'], optimizer=dogog_optim)
    discriminator.trainable = True
    discriminator.compile(loss='categorical_crossentropy', optimizer=d_optim)

    # function to get the third layer of the discriminator with a Sequential model
    get_3rd_layer_output = K.function([discriminator.layers[0].input],
                                     [discriminator.layers[9].output])

    for epoch in range(nb_epoch):
        print("Epoch is", epoch)
        print("Number of batches", int(N_train/BATCH_SIZE))
        for index in range(N_train/BATCH_SIZE):

            #training discriminator
            discriminator.trainable = True
            image0_batch, _ = get_image_batch(image0_train_gen, BATCH_SIZE)
            image1_batch, _ = get_image_batch(image1_train_gen, BATCH_SIZE)
            transformed0 = transformer0.predict(image0_batch)
            transformed1 = transformer1.predict(image1_batch)
            label_to_dis = combine_label_batch(BATCH_SIZE, BATCH_SIZE, 2*BATCH_SIZE)
            image_to_dis = combine_image_batch(image0_batch, image1_batch, transformed0, transformed1)
            d_loss = discriminator.train_on_batch(image_to_dis, label_to_dis)            
            print("discriminator batch %d d_loss : %f" % (index, d_loss))

            #training transformer0
            discriminator.trainable = False
            label_to_dual_dis0 = combine_label_batch(BATCH_SIZE, BATCH_SIZE, order='10')
            real_perceptual_features = get_3rd_layer_output([image0_batch])[0]
            image_to_dual_dis0 = image0_batch
            g_loss = dual_dis0.train_on_batch(image_to_dual_dis0, [label_to_dual_dis0, real_perceptual_features])
            print("transformer batch %d d_loss : %f" % (index, g_loss))

            #training transformer1
            label_to_dual_dis1 = combine_label_batch(BATCH_SIZE, BATCH_SIZE, order='10')
            real_perceptual_features = get_3rd_layer_output([image1_batch])[0]
            image_to_dual_dis1 = image1_batch
            g_loss = dual_dis1.train_on_batch(image_to_dual_dis1, [label_to_dual_dis1, real_perceptual_features])
            print("transformer batch %d d_loss : %f" % (index, g_loss))

            #save weights
            if index % 10 == 0:
                transformer0.save_weights('transformer0.h5')
                transformer0.save_weights('transformer1.h5')
                discriminator.save_weights('discriminator.h5')

def main():
    train(nb_epoch=10, BATCH_SIZE=5)

if __name__ == '__main__':
    main()

