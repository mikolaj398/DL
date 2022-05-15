import numpy as np
import pandas as pd
from plot import plot_roc, plot_metrics
from consts import *

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def get_image_iterator(data, image_data_generator):
    return image_data_generator.flow_from_dataframe(
        dataframe=data,
        x_col="path",
        y_col="Finding Labels",
        class_mode="categorical",
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
    )


def get_image_data_generator(augmentation=True):
    img_generator = ImageDataGenerator()

    if augmentation:
        img_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
            vertical_flip=False,
            height_shift_range=0.05,
            width_shift_range=0.1,
            rotation_range=5,
            shear_range=0.1,
            fill_mode="reflect",
            zoom_range=0.15,
        )
    return img_generator


def get_model(activation_func="relu"):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(*IMG_SIZE, 1)))
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation(activation_func))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("relu"))

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

    return model


def train(title, train_gen, test_gen, all_labels, plot):
    test_X, test_Y = next(test_gen)

    model = get_model()

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_gen,
        epochs=EPOCHS,
    )
    model_history = pd.DataFrame(history.history)
    model_history["epoch"] = history.epoch

    with open(RESULTS_PATH + f'{title}_results.json', mode='w+') as f:
        model_history.to_json(f)
    
    if plot:
        plot_metrics(title, model_history)

    return model_history
