import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from data_prep import get_combined_datasets
from consts import *
from plots import *


plot = False
if len(sys.argv) != 1:
    plot = bool(sys.argv[1])

data = get_combined_datasets()

all_labels = np.unique(data["Finding Labels"].map(lambda x: x.split("|")).tolist())
all_labels = [x for x in set(chain(*all_labels)) if len(x) > 0]

if plot:
    plot_class_count(data, all_labels)

train_df, test_df = train_test_split(data, test_size=0.25, random_state=2018)
train_df["Finding Labels"] = train_df.apply(lambda x: x["Finding Labels"].split("|"), axis=1)
test_df["Finding Labels"] = test_df.apply(lambda x: x["Finding Labels"].split("|"), axis=1)

img_data_gen = ImageDataGenerator(
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

train_gen = img_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="Finding Labels",
    class_mode="categorical",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=32,
)

if plot:
    show_image_examples(train_gen)

test_gen = img_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col="Finding Labels",
    class_mode="categorical",
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=1024,
)

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

t_x, t_y = next(train_gen)
test_X, test_Y = next(test_gen)

mobilenet_model = MobileNet(input_shape =  t_x.shape[1:], 
                                 include_top = False, weights = None)

multi_disease_model = Sequential()
multi_disease_model.add(mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=3)
callbacks_list = [checkpoint, early]

multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch=100,
                                  validation_data = (test_X, test_Y), 
                                  epochs = 10, 
                                  callbacks = callbacks_list)

for c_label, s_count in zip(all_labels, 100*np.mean(test_Y,0)):
    print('%s: %2.2f%%' % (c_label, s_count))

pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')

multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  (test_X, test_Y), 
                                  epochs = 5, 
                                  callbacks = callbacks_list)

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')                                 

