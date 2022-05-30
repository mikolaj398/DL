import pandas as pd
from plot import plot_metrics
from consts import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

def get_train_test(train_index, test_index, data, X):
    trainData = X[train_index]
    testData = X[test_index]

    train_df = data.loc[data["Image Index"].isin(list(trainData))]
    test_df = data.loc[data["Image Index"].isin(list(testData))]

    return train_df, test_df

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
            horizontal_flip=True,
            vertical_flip=False,
            height_shift_range=0.05,
            width_shift_range=0.1,
            rotation_range=15,
            fill_mode="reflect",
            zoom_range=0.15,
        )
    return img_generator


def get_model(activation_func="relu", kernel=(5,5)):
    model = Sequential()

    model.add(Conv2D(32, kernel, input_shape=(*IMG_SIZE, 1)))
    model.add(BatchNormalization())
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, kernel))
    model.add(BatchNormalization())
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, kernel))
    model.add(BatchNormalization())
    model.add(Activation(activation_func))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation(activation_func))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation(activation_func))

    model.compile(loss="binary_crossentropy", optimizer='nadam', metrics=["accuracy"])

    return model


def train(train_gen, test_gen, activation_func='relu', kernel = (3,3)):
    model = get_model(activation_func, kernel)

    history = model.fit_generator(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_gen,
        epochs=EPOCHS,
    )
    scores = model.evaluate_generator(test_gen)
    return history.history, scores[1]

def cv(name, data, X, kf, res_dict, tested_params, augmentation = True):
    fold_count = 0
    for train_index, test_index in kf.split(X):
        print(f"================================ Fold {fold_count} ================================")
        fold_count += 1
        train_df, test_df = get_train_test(train_index, test_index, data, X)

        img_data_gen = get_image_data_generator(augmentation=augmentation)
        train_gen = get_image_iterator(train_df, img_data_gen)
        test_gen = get_image_iterator(test_df, img_data_gen)

        if 'augmentation' in name:
            history, acc = train(train_gen, test_gen)
            res = {
                'fold': fold_count,
                'history': history,
                'accuracy': acc,
            }
            res_dict[name].append(res)

        else:   
            for test_param in tested_params:
                params = {name: test_param} if 'augmentation' not in name else {}
                history, acc = train(train_gen, test_gen, **params)
                res = {
                    'fold': fold_count,
                    'history': history,
                    'accuracy': acc,
                }
                res_dict[str(test_param)].append(res)