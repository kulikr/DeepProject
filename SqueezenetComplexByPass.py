import tensorflow as tf
from keras.models import Model
from keras.layers import merge,Input, Activation, Concatenate, Dropout, Convolution2D, MaxPooling2D,GlobalAveragePooling2D


def create_fire(input, s_1x1, e_1x1, e_3x3, name="fire"):
    """
    The function creates a fire module which composed of squeeze(as 's') and expand layers(as 'e')
    :param input: The input for the fire module
    :param s_1x1: The number of 1X1 filters in the squeeze layer
    :param e_1x1: The number of 1X1 filters in the expand layer
    :param e_3x3: The number of 3X3 filters in the expand layer
    :param name:  The name of the created Fire module
    :return: The created fire module
    """
    with tf.name_scope(name):
        squeeze = Convolution2D(
            s_1x1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name=name+'_squeeze')(input)
        expand_1x1 = Convolution2D(
            e_1x1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name=name+'_expand_1x1')(squeeze)
        expand_3x3 = Convolution2D(
            e_3x3, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name=name+'_expand_3x3')(squeeze)

        concatenated = Concatenate(axis=3)([expand_1x1, expand_3x3])
        return concatenated


def squeeze_net_complex_by_pass(nb_classes, inputs=(224, 224, 3)):
    """
    This function returns a an implementation of 'SqueezeNet' architecture using keras Model object
    :param nb_classes: The number of classes
    :param inputs: The dimensions of the input
    :return: A Keras Model object of SqueezeNet
    """
    input_layer = Input(shape=inputs)

    # The first convolutional layer of the squeezenet
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1')(input_layer)

    # The first max_pooling layer of the squeezenet
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    # Fire modules creation - 'first batch'
    fire2 = create_fire(input=maxpool1, s_1x1=16, e_1x1=64, e_3x3=64, name="fire2")

    # Complex ByPass - change conv1 with 1X1X128 filter
    conv1_complex = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='conv1_complex')(maxpool1)
    bypass2 = merge([conv1_complex,fire2],mode='sum')
    fire3 = create_fire(input=bypass2, s_1x1=16, e_1x1=64, e_3x3=64, name="fire3")

    # ByPass - for fire4 change the input to fire2 + fire3
    bypass3 = merge([fire2,fire3],mode='sum')
    fire4 = create_fire(input=bypass3, s_1x1=32, e_1x1=128, e_3x3=128, name="fire4")

    # Fire modules creation - 'second batch'
    # Complex ByPass - change fire3 with 1X1X256 filter
    fire3_complex = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire3_complex')(bypass3)
    bypass4 = merge([fire4,fire3_complex],mode='sum')
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool4')(bypass4)
    fire5 = create_fire(input=maxpool4, s_1x1=32, e_1x1=128, e_3x3=128, name="fire5")
    # ByPass - for fire6 change the input to maxpool4 + fire5
    bypass5 = merge([maxpool4,fire5],mode='sum')
    fire6 = create_fire(input=bypass5, s_1x1=48, e_1x1=192, e_3x3=192, name="fire6")
    # Complex ByPass - change fire6 with 1X1X384 filter
    fire6_complex = Convolution2D(
        384, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire6_complex')(fire6)
    bypass6 = merge([fire6, fire6_complex], mode='sum')
    fire7 = create_fire(input=bypass6, s_1x1=48, e_1x1=192, e_3x3=192, name="fire7")
    # ByPass - for fire8 change the input to fire6 + fire7
    bypass7 = merge([fire6,fire7],mode='sum')
    fire8 = create_fire(input=bypass7, s_1x1=64, e_1x1=256, e_3x3=256, name="fire8")

    # Complex ByPass - change fire8 with 1X1X512 filter
    fire8_complex = Convolution2D(
        512, (1, 1), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='fire8_complex')(
        fire8)
    bypass8 = merge([fire8, fire8_complex], mode='sum')
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool8')(bypass8)

    fire9 = create_fire(input=maxpool8, s_1x1=64, e_1x1=256, e_3x3=256, name="fire9")

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(fire9)
    # ByPass - for conv10 change the input to maxpool8 + fire9_dropout
    bypass9 = merge([maxpool8,fire9_dropout],mode='sum')
    conv10 = Convolution2D(nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform', name='conv10')(bypass9)

    global_avgpool10 = GlobalAveragePooling2D()(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_layer, outputs=softmax)


