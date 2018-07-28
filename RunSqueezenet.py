import keras
from Squeezenet import squeeze_net
from SqueezenetByPass import squeeze_net_by_pass
from SqueezenetComplexByPass import squeeze_net_complex_by_pass
import DataUtils as d_utils
import Alexnet
import math

# input image dimensions - from data utils
img_rows, img_cols = d_utils.IMAGE_SIZE, d_utils.IMAGE_SIZE
num_classes = d_utils.NUM_CLASSES
channels = d_utils.NUM_CHANNELS

params = {'loss': 'categorical_crossentropy',
          'base_lr': 0.001,
          'decay_lr': 0.0002,
          'momentum': 0.9,
          'epochs': 20,
          'batch_size': 30}

# Read anf prepare data
x_train, y_train, training_le = d_utils.load_training_images()

x_val, y_val = d_utils.load_validation_images(training_le)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

input_shape = (img_rows, img_cols, channels)


def run(model_name="squeezenet"):

    # Tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorboard/'+model_name, histogram_freq=0,
                                             write_graph=True, write_images=False)

    if model_name == "squeezenet":
        model = squeeze_net(num_classes, input_shape)
    elif model_name == "squeezenet_by_pass":
        model = squeeze_net_by_pass(num_classes, input_shape)
    elif model_name == "squeezenet_complex_by_pass":
        model = squeeze_net_complex_by_pass(num_classes, input_shape)
    else:
        model= Alexnet.AlexNet()

    model.compile(loss=params['loss'],
                  optimizer=keras.optimizers.SGD(lr=params['base_lr'], decay=params['decay_lr'], momentum=params['momentum']),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[tbCallBack])

    score = model.evaluate(x_val, y_val, verbose=0)
    print('Results for ' + model_name + ':')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# run('squeezenet')
# # params['base_lr'] = 0.001
# run('squeezenet_by_pass')
run('squeezenet_complex_by_pass')
# run('Alexnet')