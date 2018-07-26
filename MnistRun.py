import keras
import Main
import DataUtils as d_utils
from keras.callbacks import LearningRateScheduler
import math
from keras.applications import vgg16

# input image dimensions
img_rows, img_cols = d_utils.IMAGE_SIZE, d_utils.IMAGE_SIZE
batch_size = d_utils.BATCH_SIZE
num_classes = d_utils.NUM_CLASSES
channels = d_utils.NUM_CHANNELS
epochs = 20

params = {'loss': 'categorical_crossentropy',
          'optimizer': 'SGD',
          'base_lr': 0.04,
          'decay_lr': 0.5,
          'epochs_drop': 1.0,
          'epochs': epochs,
          'batch_size': batch_size}

# Tensorboard callback
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                         write_graph=True, write_images=False)


# Learning rate decay callback
def step_decay(epoch):
    initial_lrate = params['base_lr']
    drop = 0.5
    epochs_drop = params['epochs_drop']
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)

callbacks = [tbCallBack]

# Load the VGG model
# vgg_model = vgg16.VGG16(weights=None, input_shape=(64,64,3), classes=3)


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
model = Main.squeeze_net(num_classes, input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.001, decay=0.0002, momentum=0.9),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks)

score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
