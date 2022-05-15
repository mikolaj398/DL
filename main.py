import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split

from data_prep import get_combined_datasets
from consts import *
from plot import *
from train import *


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

img_data_gen = get_image_data_generator()
train_gen = get_image_iterator(train_df, img_data_gen)
test_gen = get_image_iterator(test_df, img_data_gen)

if plot:
    show_image_examples(train_gen)

train(train_gen, test_gen, all_labels, plot)

