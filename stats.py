from consts import RESULTS_PATH
from scipy.stats import wilcoxon, ttest_ind, ttest_rel
import scikit_posthocs as sp
import numpy as np
import json


def get_acc(aug_files, keys):
    results = {}

    for file, key in zip(aug_files, keys):
        with open(RESULTS_PATH + file, "r") as res_file:
            results[key] = json.load(res_file)[key]
    
    acc = {}
    for key, res in results.items():
        acc[key] = []
        for fold in res:
            acc[key].append(fold['history']['accuracy'][-1])
    
    return acc
with open('stats_results.txt', 'w+', encoding='UTF-8') as results_file:

    aug_files = ["aug.json", "no_aug.json"]
    keys = ['augmentation', 'no_augmentation']

    acc = get_acc(aug_files, keys)

    res = ttest_rel(acc['augmentation'], acc['no_augmentation']).pvalue
    results_file.writelines("Porówanie dokładności sieci z użyciem augmentacji i bez.\n")
    results_file.writelines(f"P-value z testu T-Studenta: {str(res)}\n\n\n")


    aug_files = ["activation_func.json", "activation_func.json", "activation_func.json"]
    keys = ['relu', 'sigmoid', 'tanh']

    acc = get_acc(aug_files, keys)

    data = np.array([acc['relu'], acc['sigmoid'], acc['tanh']])
    res = sp.posthoc_nemenyi_friedman(data.T)
    results_file.writelines("Porówanie dokładności sieci z różnymi funkcjami aktywacji\n")
    results_file.writelines("Wyniki testu Friedmana oraz analizy post-hoc z użyciem testu Nemenyi:\n")
    results_file.writelines(str(res) + '\n\n\n')

    aug_files = ["kernels.json", "kernels.json", "kernels.json", "kernels.json"]
    keys = ['(3, 3)', '(5, 5)', '(7, 7)', '(12, 12)']

    acc = get_acc(aug_files, keys)

    data = np.array([acc['(3, 3)'], acc['(5, 5)'], acc['(7, 7)'], acc['(12, 12)']])
    res = sp.posthoc_nemenyi_friedman(data.T)
    results_file.writelines("Porówanie dokładności sieci z różnymi rozmiarami splotu\n")
    results_file.writelines("Wyniki testu Friedmana oraz analizy post-hoc z użyciem testu Nemenyi:\n")
    results_file.writelines(str(res) + '\n\n\n')
