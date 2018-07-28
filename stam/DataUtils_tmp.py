import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


BATCH_SIZE = 10
NUM_CLASSES = 2
NUM_IMAGES_PER_CLASS = 183
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tmp_data\\train\\"
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
NUM_IMAGES_PER_CLASS_VAL = 50

VAL_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tmp_data\\val\\"
# VAL_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tiny imagenet\\tiny-imagenet-200\\val\\"
IMAGE_SIZE = 224
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
class_names = []

def load_training_images(images_per_class=NUM_IMAGES_PER_CLASS, dirpath=TRAINING_IMAGES_DIR):

    image_index = 0
    # init images array with the suitable sizes
    images = np.ndarray(shape=(NUM_CLASSES*images_per_class, IMAGE_ARR_SIZE))
    names = []
    labels = []
    print("Loading training images from ", dirpath)

    # Loop through all the classes directories
    for type in os.listdir(dirpath):
        class_names.append(type)

        if os.path.isdir(dirpath + type + '/images/'):
            type_images = os.listdir(dirpath + type + '/images/')

            # Loop through all the images of a type directory
            in_class_index = 0;
            for image in type_images:
                image_file = os.path.join(dirpath, type + '/images/', image)

                # load an image in PIL format
                original = load_img(image_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))

                # convert the PIL image to a numpy array
                # IN PIL - image is in (width, height, channel)
                # In Numpy - image is in (height, width, channel)
                image_data = img_to_array(original)
                # image_data = np.asarray(Image.open(image_file))

                # Validate image size is correct
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)

                    image_index += 1
                    in_class_index += 1
                if (in_class_index >= images_per_class):
                    break;

    labels = np.asarray(labels)

    shuffle_index = np.random.permutation(len(labels))
    images = images[shuffle_index]
    labels = labels[shuffle_index]

    le = preprocessing.LabelEncoder()
    training_le = le.fit(labels)
    training_labels_encoded = training_le.transform(labels)
    print("Finished loading training images.")


    return (images, training_labels_encoded, training_le)


def get_label_from_name(data, name):
    class_name = data[data['File'] == name]['Class']
    if class_name.size != 0:
        return class_name.iloc[0]
    return None


def load_validation_images(training_le, batch_size=NUM_VAL_IMAGES):
    global class_names
    print("Loading validation images from ", VAL_IMAGES_DIR)

    validation_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None,
                           names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    labels = []
    names = []
    image_index = 0

    images = np.ndarray(shape=(NUM_IMAGES_PER_CLASS_VAL*NUM_CLASSES, IMAGE_ARR_SIZE))
    val_images = os.listdir(VAL_IMAGES_DIR + '/images/')

    # Loop through all the images of a val directory
    batch_index = 0

    for image in val_images:
        image_file = os.path.join(VAL_IMAGES_DIR, 'images/', image)
        label = get_label_from_name(validation_data, image)
        if label not in class_names:
            continue

        # load an image in PIL format
        original = load_img(image_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))

        # convert the PIL image to a numpy array
        # IN PIL - image is in (width, height, channel)
        # In Numpy - image is in (height, width, channel)
        image_data = img_to_array(original)
        # reading the images as they are; no normalization, no color editing
        # image_data = np.asarray(Image.open(image_file))
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            print("Image Num :" + str(image_index))
            image_index += 1
            labels.append(label)
            names.append(image)
            batch_index += 1

        if (batch_index >= batch_size):
            break

    labels = np.asarray(labels)

    val_labels_encoded = training_le.transform(labels)
    print("Finished loading validation images.")

    return (images, val_labels_encoded)
