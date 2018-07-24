import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
from sklearn import preprocessing


BATCH_SIZE = 20
NUM_CLASSES = 3
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tiny imagenet\\tiny-imagenet-200\\train\\"
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 9832
VAL_IMAGES_DIR = "C:\\Users\\royku\\Desktop\\tiny imagenet\\tiny-imagenet-200\\val\\"
IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS


def load_training_images(images_per_class=500):
    image_index = 0
    # init images array with the suitable sizes
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []
    print("Loading training images from ", TRAINING_IMAGES_DIR)

    i=0
    # Loop through all the classes directories
    for type in os.listdir(TRAINING_IMAGES_DIR):
        if i == NUM_CLASSES:
            break
        i+=1
        if os.path.isdir(TRAINING_IMAGES_DIR + type + '/images/'):
            type_images = os.listdir(TRAINING_IMAGES_DIR + type + '/images/')

            # Loop through all the images of a type directory
            in_class_index = 0;
            for image in type_images:
                image_file = os.path.join(TRAINING_IMAGES_DIR, type + '/images/', image)

                image_data = np.asarray(Image.open(image_file))

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
    for idx, row in data.iterrows():
        if (row['File'] == name):
            return row['Class']

    return None


def load_validation_images(training_le, batch_size=5):
    print("Loading validation images from ", VAL_IMAGES_DIR)

    validation_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None,
                           names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    labels = []
    names = []
    image_index = 0

    images = np.ndarray(shape=(batch_size, IMAGE_ARR_SIZE))
    val_images = os.listdir(VAL_IMAGES_DIR + '/images/')

    # Loop through all the images of a val directory
    batch_index = 0;

    for image in val_images:
        image_file = os.path.join(VAL_IMAGES_DIR, 'images/', image)

        # reading the images as they are; no normalization, no color editing
        image_data = np.asarray(Image.open(image_file))
        if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
            images[image_index, :] = image_data.flatten()
            image_index += 1
            labels.append(get_label_from_name(validation_data, image))
            names.append(image)
            batch_index += 1

        if (batch_index >= batch_size):
            break;

    labels = np.asarray(labels)

    val_labels_encoded = training_le.transform(labels)
    print("Finished loading validation images.")

    return (images, val_labels_encoded)
