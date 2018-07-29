import keras
from TinySqueezenet import tiny_squeeze_net
import DataUtils as d_utils

# Models parameters
params = {'loss': 'categorical_crossentropy',
          'base_lr': 0.02,
          'decay_lr': 0.0002,
          'momentum': 0.9,
          'epochs': 300,
          'batch_size': 1000}


def run(model_name="tiny_squeezenet"):
    # input image dimensions - from data utils
    img_rows, img_cols = d_utils.IMAGE_SIZE, d_utils.IMAGE_SIZE
    num_classes = d_utils.NUM_CLASSES
    channels = d_utils.NUM_CHANNELS

    # Read and prepare data
    x_train, y_train, training_le = d_utils.load_training_images()

    x_test, y_test = d_utils.load_test_images(training_le)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = (img_rows, img_cols, channels)

    # Tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tensorboard/' + model_name, histogram_freq=0,
                                             write_graph=True, write_images=False)

    # Build the model
    model = tiny_squeeze_net(num_classes, input_shape)

    # Compile the model
    model.compile(loss=params['loss'],
                  optimizer=keras.optimizers.SGD(lr=params['base_lr'], decay=params['decay_lr'],
                                                 momentum=params['momentum']), metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack])

    # Print Results
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Results for ' + model_name + ':')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


run()
