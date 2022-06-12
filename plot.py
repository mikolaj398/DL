from calendar import EPOCH
import matplotlib.pyplot as plt
from consts import EPOCHS, PLOTS_PATH, RESULTS_PATH, FOLDS
import numpy as np
import json


def cmp_avgs(files, keys, title):
    for file, key in zip(files, keys):
        with open(RESULTS_PATH + file, 'r') as res_file:
            results = json.load(res_file)[key]
            results_matrix = np.zeros((FOLDS, EPOCHS))
            for fold in results:
                fold_results = fold['history']['accuracy']
                results_matrix[fold["fold"] -1] = fold_results
            
            avg = np.average(results_matrix, axis=0)
            plt.plot(np.arange(0, EPOCHS), avg, label=f'{key} folds avg')
    plt.title(title.title())
    plt.legend()
    plt.savefig(PLOTS_PATH + title.replace(' ', '_')+'.png')
    plt.cla()
    plt.clf()
    plt.show()

def plot_folds(file_name, data_key):
    METRICS = ['accuracy', 'loss']
    
    with open(RESULTS_PATH + file_name, 'r') as res_file:
        results = json.load(res_file)[data_key]
        for metric in METRICS:
            fig, axs = plt.subplots(2)
            suptitle = fig.suptitle(data_key.replace('_', ' ').title() + ' ' + metric, fontsize=16)
            results_matrix = np.zeros((FOLDS, EPOCHS))
            for fold in results:
                fold_results = fold['history'][metric]
                results_matrix[fold["fold"] -1] = fold_results
                axs[0].plot(np.arange(0, EPOCHS), fold_results, label=f'Fold {fold["fold"]}')
            axs[0].legend()

            avg = np.average(results_matrix, axis=0)
            axs[1].plot(np.arange(0, EPOCHS), avg, label=f'All folds avg')
            axs[1].legend()
            fig.savefig(PLOTS_PATH + f"{data_key}_{metric}.png")
            plt.cla()
            plt.clf()
            plt.show()

plot_folds('no_aug.json', 'no_augmentation')
plot_folds('aug.json', 'augmentation')
cmp_avgs(['no_aug.json', 'aug.json'], ['no_augmentation', 'augmentation'], "Augmentation vs No Augmentation")
plot_folds('activation_func.json', 'relu')
plot_folds('activation_func.json', 'sigmoid')
plot_folds('activation_func.json', 'tanh')
cmp_avgs(['activation_func.json', 'activation_func.json', 'activation_func.json'], ['relu', 'sigmoid', 'tanh'], "Activation Functions")
plot_folds('kernels.json', '(3, 3)')
plot_folds('kernels.json', '(5, 5)')
plot_folds('kernels.json', '(7, 7)')
plot_folds('kernels.json', '(12, 12)')
cmp_avgs(['kernels.json', 'kernels.json', 'kernels.json', 'kernels.json'], 
            ['(3, 3)', '(5, 5)', '(7, 7)', '(12, 12)'], "Kernels")