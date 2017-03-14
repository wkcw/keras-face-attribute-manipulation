from keras.preprocessing.image import ImageDataGenerator
import numpy as np
img_size = 128

def get_gen(class_dir, BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=False)
    train_gen = train_datagen.flow_from_directory(
        "data/%s"%class_dir,
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode="categorical")
    return train_gen

def get_image_batch(generator, BATCH_SIZE):
    """keras generators may generate an incomplete batch for the last batch"""
    img_batch = generator.next()
    if len(img_batch[0]) != BATCH_SIZE:
        img_batch = generator.next()

    assert len(img_batch[0]) == BATCH_SIZE

    return img_batch

def combine_image_batch(class0_batch, class1_batch, transformed0_batch, transformed1_batch):
    discriminator_train_batch = np.concatenate((class0_batch, class1_batch, transformed0_batch, transformed1_batch))
    return discriminator_train_batch

def combine_label_batch(num0, num1, numt=0, order='01'):
    assert order=='01' or order=='10'
    label_batch_0 = np.tile((1,0,0),(num0,1))
    label_batch_1 = np.tile((0,1,0),(num1,1))
    label_batch_t = np.tile((0,0,1),(numt,1))
    if order == '01':
        label_batch_all = np.row_stack((label_batch_0, label_batch_1, label_batch_t))
    else:
        label_batch_all = np.row_stack((label_batch_1, label_batch_0, label_batch_t))
    label_batch_all = label_batch_all.astype('float32')
    return label_batch_all