from random import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_curve, auc
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
all_labels = np.unique(data["Finding Labels"])

if plot:
    plot_class_count('All data', data, all_labels)

train_df, test_df = train_test_split(data, test_size=0.25, stratify=data["Finding Labels"], random_state=42)

if plot:
    plot_class_count('Train data', train_df, all_labels)
    plot_class_count('Test data', test_df, all_labels)

# ================================ Experiment 1 ================================
# print("================================ Experiment 1 ================================")
# results = []

# img_data_gen = get_image_data_generator(augmentation=False)
# train_gen = get_image_iterator(train_df, img_data_gen)
# test_gen = get_image_iterator(test_df, img_data_gen)
# results.append(train('no_augmentation', train_gen, test_gen, plot))

# if plot:
#     show_image_examples(train_gen)

# img_data_gen = get_image_data_generator(augmentation=True)
# train_gen = get_image_iterator(train_df, img_data_gen)
# test_gen = get_image_iterator(test_df, img_data_gen)
# results.append(train('augmentation', train_gen, test_gen, plot))

# if plot:
#     plot_augmentation(results)

# ================================ Experiment 2 ================================
# print("================================ Experiment 2 ================================")
# activation_funcs = ['relu', 'sigmoid', 'tanh']
# results = []

# img_data_gen = get_image_data_generator(augmentation=True)
# train_gen = get_image_iterator(train_df, img_data_gen)
# test_gen = get_image_iterator(test_df, img_data_gen)

# for activation_func in activation_funcs:
#     results.append(train(activation_func, train_gen, test_gen, plot, activation_func=activation_func))

# if plot:
#     plot_activations_funcs(results)

# ================================ Experiment 3 ================================
print("================================ Experiment 3 ================================")
kernels = [(3,3), (5,5), (7,7), (12, 12)]
results = []

img_data_gen = get_image_data_generator(augmentation=True)
train_gen = get_image_iterator(train_df, img_data_gen)
test_gen = get_image_iterator(test_df, img_data_gen)

for kernel in kernels:
    results.append(train(f'kernel_size_{kernel[0]}x{kernel[1]}', train_gen, test_gen, plot, kernel=kernel))

if plot:
    plot_kernels(results)