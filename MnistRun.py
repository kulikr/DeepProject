import keras
from keras.datasets import mnist,cifar10
from keras.utils import plot_model
import Main
import DataUtils as d_utils


# input image dimensions
img_rows, img_cols = d_utils.IMAGE_SIZE, d_utils.IMAGE_SIZE
batch_size = d_utils.BATCH_SIZE
num_classes = d_utils.NUM_CLASSES
channels = d_utils.NUM_CHANNELS
epochs = 5

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=False)

x_train , y_train, training_le = d_utils.load_training_images()

x_val , y_val = d_utils.load_validation_images(training_le)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, channels)

x_train = x_train.astype('float32')
x_test = x_val.astype('float32')

x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_val, num_classes)

model = Main.squeeze_net(num_classes,(img_rows,img_cols,channels))
# plot_model(model,'model.png')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[tbCallBack])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])