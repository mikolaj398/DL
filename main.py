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
    batch_size=8,
)

if plot:
    show_image_examples(train_gen)
