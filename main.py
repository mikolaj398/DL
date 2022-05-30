
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold

from data_prep import get_combined_datasets
from consts import *
from train import cv
import json

data = get_combined_datasets()
all_labels = np.unique(data["Finding Labels"])

kf = KFold(n_splits=FOLDS, random_state=42, shuffle=True)
X = np.array(data["Image Index"])

print("================================ Experiment 1 ================================")
results = {
    'no_augmentation': [],
}
cv('no_augmentation', data, X, kf, results, [], augmentation = False)
with open(RESULTS_PATH + 'no_aug.json', 'w+') as res_file:
    json.dump(results, res_file)

results = {
    'augmentation': []
}
cv('augmentation', data, X, kf, results, [], augmentation = True)
with open(RESULTS_PATH + 'aug.json', 'w+') as res_file:
    json.dump(results, res_file)


print("================================ Experiment 2 ================================")
activation_funcs = ['relu', 'sigmoid', 'tanh']
results = {
    'relu': [],
    'sigmoid': [],
    'tanh': [],
}
cv('activation_func', data, X, kf, results, activation_funcs, augmentation = True)
with open(RESULTS_PATH + 'activation_func.json', 'w+') as res_file:
    json.dump(results, res_file)

print("================================ Experiment 3 ================================")
kernels = [(3,3), (5,5), (7,7), (12, 12)]
results = {
    '(3, 3)': [],
    '(5, 5)': [],
    '(7, 7)': [],
    '(12, 12)': [],
}
cv('kernel', data, X, kf, results, kernels, augmentation = True)
with open(RESULTS_PATH +  'kernels.json', 'w+') as res_file:
    json.dump(results, res_file)