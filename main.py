import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from data_prep import get_combined_datasets
from consts import *

data = get_combined_datasets()

all_labels = np.unique(data["Finding Labels"].map(lambda x: x.split("|")).tolist())
all_labels = [x for x in set(chain(*all_labels)) if len(x) > 0]

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

all_labels = list(train_gen.class_indices.keys())

iter_image, iter_labels = next(train_gen)
fig, ax = plt.subplots(2, 4)
for image, labels, plot in zip(iter_image, iter_labels, ax.flatten()):
    plot.imshow(image[:, :, 0], cmap="bone")
    plot.set_title(
        ", ".join([class_name for class_name, belongs in zip(all_labels, labels) if belongs == 1])
    )
    plot.axis("off")

plt.show()
