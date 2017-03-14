from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.regularizers import l1, activity_l1
from keras.utils.visualize_util import plot
from utils import *




def transform_model(weight_loss_pix=5e-4):
    inputs = Input(shape=( 128, 128, 3))
    x1 = Convolution2D(64, 5, 5, border_mode='same')(inputs)
    x2 = LeakyReLU(alpha=0.3, name='wkcw')(x1)
    x3 = BatchNormalization()(x2)
    x4 = Convolution2D(128, 4, 4, border_mode='same', subsample=(2,2))(x3)
    x5 = LeakyReLU(alpha=0.3)(x4)
    x6 = BatchNormalization()(x5)
    x7 = Convolution2D(256, 4, 4, border_mode='same', subsample=(2,2))(x6)
    x8 = LeakyReLU(alpha=0.3)(x7)
    x9 = BatchNormalization()(x8)
    x10 = Deconvolution2D(128, 3, 3, output_shape=(None, 64, 64, 128), border_mode='same', subsample=(2,2))(x9)
    x11 = BatchNormalization()(x10)
    x12 = Deconvolution2D(64, 3, 3, output_shape=(None, 128, 128, 64), border_mode='same', subsample=(2,2))(x11)
    x13 = BatchNormalization()(x12)
    x14 = Deconvolution2D(3, 4, 4, output_shape=(None, 128, 128, 3), border_mode='same', activity_regularizer=activity_l1(weight_loss_pix))(x13)
    output = merge([inputs, x14], mode='sum')
    model = Model(input=inputs, output=output)

    return model

def discriminator_both():

    inputs = Input(shape=( 128, 128, 3))
    x1 = Convolution2D(64, 4, 4, border_mode='same', subsample=(2,2), input_shape=(128, 128, 3))(inputs)
    x2 = LeakyReLU(alpha=0.3)(x1)
    x3 = BatchNormalization()(x2)
    x4 = Convolution2D(128, 4, 4, border_mode='same', subsample=(2,2))(x3)
    x5 = LeakyReLU(alpha=0.3)(x4)
    x6 = BatchNormalization()(x5)
    x7 = Convolution2D(256, 4, 4, border_mode='same', subsample=(2,2))(x6)
    x8 = LeakyReLU(alpha=0.3)(x7)
    output2 = BatchNormalization(name='perceptual')(x8)
    x10 = Convolution2D(512, 4, 4, border_mode='same', subsample=(2,2))(output2)
    x11 = LeakyReLU(alpha=0.3)(x10)
    x12 = BatchNormalization()(x11)
    x13 = Convolution2D(1024, 4, 4, border_mode='same', subsample=(2,2))(x12)
    x14 = LeakyReLU(alpha=0.3)(x13)
    x15 = BatchNormalization()(x14)
    x16 = Convolution2D(1, 4, 4, border_mode='same', subsample=(2,2))(x15)
    x17 = LeakyReLU(alpha=0.3)(x16)
    x18 = BatchNormalization()(x17)
    x19 = Flatten()(x18)
    x20 = Dense(3)(x19)
    output = Activation('softmax', name='softmax')(x20)
    model_1 = Model(input=inputs, output=output)
    model_2 = Model(input=inputs, output=output2)
    return model_1, model_2

def dual_dis(transformer0, transformer1, discriminator, discriminator3rd):
    make_trainable(discriminator,False)
    dual_image = transformer1(transformer0.outputs[0])
    mid_image = transformer0.outputs[0]
    mid_score = discriminator(mid_image)
    dual_score = discriminator(dual_image)
    features = discriminator3rd(mid_image)
    model = Model(input=[transformer0.inputs[0]], output=[mid_score, features, dual_score])
    return model

# Changes the traiable argument for all the layers of model
# to the boolean argument "trainable"
def make_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable

def train(nb_epoch=100, BATCH_SIZE=32, weight_loss_pix=5e-4, weight_loss_per=5e-5):
    N_train = 10
    image0_train_gen = get_gen('no', BATCH_SIZE)
    image1_train_gen = get_gen('with', BATCH_SIZE)
    d_optim = RMSprop(lr=0.001, decay=0.001)
    #g_optim = RMSprop(lr=0.001, decay=0.001)
    dogog_optim = RMSprop(lr=0.001, decay=0.001)

    discriminator, discriminator3rd = discriminator_both()
    discriminator.compile(loss='categorical_crossentropy', optimizer=d_optim)
    discriminator._make_train_function()

    transformer0 = transform_model(weight_loss_pix)
    transformer1 = transform_model(weight_loss_pix)

    dual_dis0 = dual_dis(transformer0, transformer1, discriminator, discriminator3rd)
    dual_dis1 = dual_dis(transformer1, transformer0, discriminator, discriminator3rd)
    dual_dis0.compile(loss=[ "categorical_crossentropy", "mse", "categorical_crossentropy"], loss_weights=[1., weight_loss_per, 1.], optimizer=dogog_optim)
    dual_dis1.compile(loss=[ "categorical_crossentropy", "mse", "categorical_crossentropy"], loss_weights=[1., weight_loss_per, 1.], optimizer=dogog_optim)


    # function to get the third layer of the discriminator with a Sequential model
    get_3rd_layer_output = K.function([discriminator.layers[0].input],[discriminator.layers[8].output])

    for epoch in range(nb_epoch):
        print("Epoch is", epoch)
        print("Number of batches", int(N_train/BATCH_SIZE))
        for index in range(int(N_train/BATCH_SIZE)):

            #training discriminator
            make_trainable(discriminator, True)
            make_trainable(discriminator3rd, True)
            image0_batch, _ = get_image_batch(image0_train_gen, BATCH_SIZE)
            image1_batch, _ = get_image_batch(image1_train_gen, BATCH_SIZE)
            transformed0 = transformer0.predict(image0_batch)
            transformed1 = transformer1.predict(image1_batch)
            label_to_dis = combine_label_batch(BATCH_SIZE, BATCH_SIZE, 2*BATCH_SIZE)
            image_to_dis = combine_image_batch(image0_batch, image1_batch, transformed0, transformed1)
            d_loss = discriminator.train_on_batch(image_to_dis, label_to_dis)            
            print("discriminator batch %d d_loss : %f" % (index, d_loss))

            #training transformer0
            #discriminator.trainable = False
            make_trainable(discriminator,False)
            make_trainable(discriminator3rd, False)
            label1_to_dual_dis0 = combine_label_batch(0, BATCH_SIZE, order='10')#(BATCH_SIZE, BATCH_SIZE, order='10')
            label0_to_dual_dis0 = combine_label_batch(BATCH_SIZE, 0, order='10')
            real_perceptual_features = discriminator3rd.predict(image0_batch)
            #real_perceptual_features = get_3rd_layer_output([image0_batch])[0]
            image_to_dual_dis0 = image0_batch
            g_loss = dual_dis0.train_on_batch(image_to_dual_dis0, [label1_to_dual_dis0, real_perceptual_features, label0_to_dual_dis0])
            print("dual_dis0 batch %d g_loss : %f" % (index, g_loss[0]))

            #training transformer1
            label0_to_dual_dis1 = combine_label_batch(BATCH_SIZE, 0, order='01')
            label1_to_dual_dis1 = combine_label_batch(BATCH_SIZE, 0, order='01')
            real_perceptual_features = discriminator3rd.predict(image1_batch)#get_3rd_layer_output([image0_batch])[0]
            image_to_dual_dis1 = image1_batch
            g_loss = dual_dis1.train_on_batch(image_to_dual_dis1, [label0_to_dual_dis1, real_perceptual_features, label1_to_dual_dis1])
            print("dual_dis1 batch %d g_loss : %f" % (index, g_loss[0]))

            #save weights
            if index % 10 == 0:
                transformer0.save_weights('transformer0.h5')
                transformer0.save_weights('transformer1.h5')
                discriminator.save_weights('discriminator.h5')

def main():
    train(nb_epoch=1, BATCH_SIZE=5, weight_loss_pix=5e-4, weight_loss_per=5e-5)

if __name__ == '__main__':
    main()

