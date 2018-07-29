import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

################# Parameters for the data ##########################

NUM_CLASSES = 200

NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = "./data/tiny-imagenet-200/train/"
TRAIN_SIZE = NUM_IMAGES

NUM_IMAGES_PER_CLASS_TEST = 50
TEST_IMAGES_DIR = "./data/tiny-imagenet-200/test/"
TEST_SIZE = NUM_CLASSES * NUM_IMAGES_PER_CLASS_TEST

IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
class_names = []


#####################################################################

def load_training_images():
    global class_names

    image_index = 0

    # init images array with the suitable sizes
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []
    print("Loading train images from ", TRAINING_IMAGES_DIR)

    i = 0
    # Loop through all the classes directories
    for type in os.listdir(TRAINING_IMAGES_DIR):
        if i == NUM_CLASSES:
            break
        i += 1
        class_names.append(type)

        if os.path.isdir(TRAINING_IMAGES_DIR + type + '/images/'):
            type_images = os.listdir(TRAINING_IMAGES_DIR + type + '/images/')

            # Loop through all the images of a type directory
            in_class_index = 0
            for image in type_images:
                image_file = os.path.join(TRAINING_IMAGES_DIR, type + '/images/', image)

                # load an image in PIL format
                original = load_img(image_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))

                image_data = img_to_array(original)

                # Validate image size is correct
                if image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)

                    image_index += 1
                    in_class_index += 1
                if in_class_index >= NUM_IMAGES_PER_CLASS:
                    break

    labels = np.asarray(labels)

    shuffle_index = np.random.permutation(len(labels))
    images = images[shuffle_index]
    labels = labels[shuffle_index]

    le = preprocessing.LabelEncoder()
    training_le = le.fit(labels)
    training_labels_encoded = training_le.transform(labels)
    print("Finished loading train images.")

    return images, training_labels_encoded, training_le


def get_label_from_name(meta_data, name):
    class_name = meta_data[meta_data['File'] == name]['Class']
    if class_name.size != 0:
        return class_name.iloc[0]
    return None


def load_test_images(training_le):
    global class_names
    print("Loading test images from ", TEST_IMAGES_DIR)

    test_meta_data = pd.read_csv(TEST_IMAGES_DIR + 'test_annotations.txt', sep='\t', header=None,
                                 names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    labels = []
    names = []

    images = np.ndarray(shape=(NUM_IMAGES_PER_CLASS_TEST * NUM_CLASSES, IMAGE_ARR_SIZE))
    test_images = os.listdir(TEST_IMAGES_DIR + '/images/')

    # Loop through all the images of a val directory
    image_index = 0

    for image in test_images:
        image_file = os.path.join(TEST_IMAGES_DIR, 'images/', image)
        label = get_label_from_name(test_meta_data, image)
        if label not in class_names:
            continue

        # load an image
        original = load_img(image_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))

        image_data = img_to_array(original)

        if image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
            images[image_index, :] = image_data.flatten()
            labels.append(label)
            names.append(image)
            image_index += 1

        if image_index >= TEST_SIZE:
            break

    labels = np.asarray(labels)

    test_labels_encoded = training_le.transform(labels)
    print("Finished loading test images.")

    return images, test_labels_encoded
