from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
import csv
import os
import sklearn
import cv2
import numpy as np


def preprocess_pipeline (data_path='./data/', correction=.18):
    """
    Go over the data directory
    look for subdirectories that have a 'driving_log.csv' file inside
    each one is a data folder
    load the image names and the measurements
    out of each data create a log for the 'left', 'right' and 'center'
    images
    :param data_path: where to look for the data
    :param correction: what is the angle correction for the 'left' and 'right' images
    :return: tuple (images, measurments)
    """
    directories = [x[0] for x in os.walk(data_path)]
    data_dirs = list(filter(lambda dir: os.path.isfile(
        os.path.join(dir, 'driving_log.csv')
    ), directories))

    for data_dir in data_dirs:
        # read all the lines from the csv driving_log file
        with open(os.path.join(data_dir, 'driving_log.csv'), 'r') as f:
            reader = csv.reader(f)
            csv_lines = list(reader)

        centers = []
        lefts = []
        rights = []
        measurements = []
        for line in csv_lines:

            centers.append(
                os.path.join(data_dir, 'IMG', os.path.basename(line[0]))
            )
            lefts.append(
                os.path.join(data_dir, 'IMG', os.path.basename(line[1]))
            )
            rights.append(
                os.path.join(data_dir, 'IMG', os.path.basename(line[2]))
            )

            measurements.append(float(line[3]))

        features_filenames = []
        features_filenames.extend(centers)
        features_filenames.extend(lefts)
        features_filenames.extend(rights)
        targets = []
        targets.extend(measurements)
        targets.extend([x + correction for x in measurements])
        targets.extend([x - correction for x in measurements])

        return features_filenames, targets


def get_batches(samples, batch_size=1024):
    """
    Generator function to yield batches for the training
    :param samples: data samples
    :param batch_size: batches size to yield
    :return: a batch_size out of the samples data
    """
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]

            features = []
            targets = []
            # create the batch
            for features_filename, target in batch_samples:
                img = cv2.imread(features_filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # append the original image
                features.append(img)
                targets.append(target)

                # fip th image and reverse the angle
                features.append(cv2.flip(img,1))
                targets.append(-target)

            inputs = np.asarray(features)
            outputs = np.asarray(targets)
            # yield the batch
            yield sklearn.utils.shuffle(inputs, outputs)


def build_classifier_nvidia():
    """
    Create the NVidea model
    :return: keras model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
