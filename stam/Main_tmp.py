import keras
import Squeezenet
import DataUtils_tmp as d_utils
from keras.applications import vgg16

# input image dimensions
img_rows, img_cols = d_utils.IMAGE_SIZE, d_utils.IMAGE_SIZE
batch_size = d_utils.BATCH_SIZE
num_classes = d_utils.NUM_CLASSES
channels = d_utils.NUM_CHANNELS
epochs = 1000

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=False)


# Load the VGG model
vgg_model = vgg16.VGG16(weights=None, classes=3)
TRAINING_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tmp_data\\train\\"
VAL_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tmp_data\\val\\"

x_train , y_train, training_le = d_utils.load_training_images()

x_val , y_val,_ = d_utils.load_training_images(26,VAL_IMAGES_DIR)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, channels)



x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

x_train /= 255
x_val /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


model = Squeezenet.squeeze_net(num_classes, input_shape)
# model = vgg_model

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adagrad(lr=0.05),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[tbCallBack])
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])