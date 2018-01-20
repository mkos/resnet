import tensorflow as tf
#from tensorflow import keras
import keras
import keras.layers as layers
import keras.models as models
import keras.regularizers as regularizers
import keras.preprocessing.image as image_preproc
import keras.backend as K
import numpy as np

from dataio import history_to_json, maybe_train_test_split

print('tensorflow', tf.__version__, 'keras', keras.__version__)

def residual_block(input_tensor, filters, layer_num, reg, downsample=False):
    """
    Residual block with bottleneck layers and preactivation
    :param input_tensor: the input
    :param filters: tuple of three channel numbers for member conv layers
    :param layer_num: # of this layer
    :param reg: regularization parameter
    :param downsample: reduce size of output by 2
    :return: output tensor of the same shape that input_tensor or with dimensions reduced by two if reduce=True
    """
    bn_name = lambda ver: 'bn_{}_{}'.format(layer_num, ver)
    act_name = lambda ver: 'act_{}_{}'.format(layer_num, ver)
    conv_name = lambda ver: 'conv_{}_{}'.format(layer_num, ver)
    merge_name = 'merge_{}'.format(layer_num)

    if layer_num > 1:
        x = layers.BatchNormalization(name=bn_name('a'))(input_tensor)
        x = layers.Activation('relu', name=act_name('a'))(x)
    else:
        x = input_tensor

    x = layers.Conv2D(filters[0], (1, 1),
                      use_bias=False,
                      padding='same',
                      kernel_regularizer=regularizers.l2(reg),
                      name=conv_name('a'))(x)

    x = layers.BatchNormalization(name=bn_name('b'))(x)
    x = layers.Activation('relu', name=act_name('b'))(x)
    x = layers.Conv2D(filters[1], (3, 3),
                      use_bias=False,
                      padding='same',
                      kernel_regularizer=regularizers.l2(reg),
                      name=conv_name('b'))(x)

    x = layers.BatchNormalization(name=bn_name('c'))(x)
    x = layers.Activation('relu', name=act_name('c'))(x)

    final_stride = (2,2) if downsample else (1,1)
    x = layers.Conv2D(filters[2], (1,1),
                      strides=final_stride,
                      use_bias=False,
                      padding='same',
                      kernel_regularizer=regularizers.l2(reg),
                      name=conv_name('c'))(x)

    input_tensor = expand_channels_bottleneck(input_tensor, filters[2], layer_num) if downsample else input_tensor
    x = layers.add([input_tensor, x], name=merge_name)
    return x

def expand_channels_bottleneck(tensor, new_channels, layer_num):
    N, H, W, C = K.int_shape(tensor)
    assert new_channels - C > 0
    return layers.Conv2D(new_channels, (1,1), strides=(2,2), name='resize_{}'.format(layer_num))(tensor)

def resnet_model(input_shape, num_classes, num_layers, reg):

    input_tensor = layers.Input(shape=input_shape)

    x = layers.Conv2D(128, (3, 3),
                      use_bias=False,
                      padding='same',
                      kernel_regularizer=regularizers.l2(reg),
                      name='conv_0')(input_tensor)
    x = layers.BatchNormalization(name='bn_0')(x)
    x = layers.Activation('relu', name='relu_0')(x)

    x = residual_block(x, [32, 32, 128], layer_num=1, reg=reg)
    x = residual_block(x, [32, 32, 128], layer_num=2, reg=reg)
    x = residual_block(x, [32, 32, 256], layer_num=3, reg=reg, downsample=True)
    x = residual_block(x, [16, 16, 256], layer_num=4, reg=reg)
    x = residual_block(x, [16, 16, 256], layer_num=5, reg=reg)

    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.Activation('relu', name='relu_final')(x)

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc_final')(x)

    return models.Model(input_tensor, x)


def main():
    from datetime import datetime as dt
    start = dt.now()
    model = resnet_model(input_shape=(32, 32, 3), num_classes=5, num_layers=5, reg=0.0)
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # flowers dataset from: http://download.tensorflow.org/example_images/flower_photos.tgz
    train_dir, test_dir = maybe_train_test_split('/Users/kosa/repos/datasets/flowers')

    # create dataset generators
    train_generator = (image_preproc
                       .ImageDataGenerator(rescale=1./255, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
                       .flow_from_directory(train_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')
                       )

    test_generator = (image_preproc
                      .ImageDataGenerator(rescale=1./255)
                      .flow_from_directory(test_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')
                      )

    print('train classes:', train_generator.class_indices)
    print('test classes:', test_generator.class_indices)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=test_generator,
        validation_steps=20)

    runtime = dt.now() - start
    print('Training output file:', history_to_json(history.history, str(runtime)))
    print('Time spent:', runtime)

if __name__ == '__main__':
    main()

