
from keras.models import Model
from keras.layers import Input, Activation, Concatenate, Dropout, Convolution2D, MaxPooling2D,GlobalAveragePooling2D


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

    squeeze = Convolution2D(
        s_1x1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name=name+'_squeeze')(input)
    expand_1x1 = Convolution2D(
        e_1x1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name=name+'_expand_1x1')(squeeze)
    expand_3x3 = Convolution2D(
        e_3x3, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name=name+'_expand_3x3')(squeeze)

    return Concatenate(axis=1)([expand_1x1, expand_3x3])


def squeeze_net(nb_classes, inputs=(224, 224, 3)):
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
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first")(conv1)

    # Fire modules creation - 'first batch'
    fire2 = create_fire(input=maxpool1, s_1x1=16, e_1x1=64, e_3x3=64, name="fire2")
    fire3 = create_fire(input=fire2, s_1x1=16, e_1x1=64, e_3x3=64, name="fire3")
    fire4 = create_fire(input=fire3, s_1x1=32, e_1x1=128, e_3x3=128, name="fire4")

    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(fire4)

    # Fire modules creation - 'second batch'
    fire5 = create_fire(input=fire4, s_1x1=32, e_1x1=128, e_3x3=128, name="fire5")
    fire6 = create_fire(input=fire5, s_1x1=48, e_1x1=192, e_3x3=192, name="fire6")
    fire7 = create_fire(input=fire6, s_1x1=48, e_1x1=192, e_3x3=192, name="fire7")
    fire8 = create_fire(input=fire7, s_1x1=64, e_1x1=256, e_3x3=256, name="fire8")

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(fire8)

    fire9 = create_fire(input=maxpool8, s_1x1=64, e_1x1=256, e_3x3=256, name="fire9")

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(fire9)

    conv10 = Convolution2D(
        nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform')(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D()(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    return Model(inputs=input_layer, outputs=softmax)


