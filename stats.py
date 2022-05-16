from consts import RESULTS_PATH
from scipy.stats import wilcoxon, ttest_ind
import scikit_posthocs as sp
import numpy as np
import json

with open('stats_results.txt', 'w+', encoding='UTF-8') as results_file:

    aug_files = ["augmentation_results.json", "no_augmentation_results.json"]
    results = {}

    for file in aug_files:
        with open(RESULTS_PATH + file, "r") as res_file:
            results[file.replace("_results.json", "")] = json.load(res_file)["accuracy"]

    aug_socres = [val for val in results["augmentation"].values()]
    no_aug_socres = [val for val in results["no_augmentation"].values()]

    res = ttest_ind(aug_socres, no_aug_socres).pvalue
    results_file.writelines("Porówanie dokładności sieci z użyciem augmentacji i bez.\n")
    results_file.writelines(f"P-value z testu T-Studenta: {str(res)}\n\n\n")

    activ_func_files = ["relu_results.json", "sigmoid_results.json", "tanh_results.json"]
    results = {}

    for file in activ_func_files:
        with open(RESULTS_PATH + file, "r") as res_file:
            results[file.replace("_results.json", "")] = json.load(res_file)["accuracy"]

    relu_socres = [val for val in results["relu"].values()]
    sigmoid_socres = [val for val in results["sigmoid"].values()]
    tanh_socres = [val for val in results["tanh"].values()]

    data = np.array([relu_socres, sigmoid_socres, tanh_socres])
    res = sp.posthoc_nemenyi_friedman(data.T)
    results_file.writelines("Porówanie dokładności sieci z różnymi funkcjami aktywacji\n")
    results_file.writelines("Wyniki testu Friedmana oraz analizy post-hoc z użyciem testu Nemenyi:\n")
    results_file.writelines(str(res) + '\n\n\n')

    kernels_files = [
        "kernel_size_3x3_results.json",
        "kernel_size_5x5_results.json",
        "kernel_size_7x7_results.json",
        "kernel_size_12x12_results.json",
    ]
    results = {}

    for file in kernels_files:
        with open(RESULTS_PATH + file, "r") as res_file:
            results[file.replace("_results.json", "")] = json.load(res_file)["accuracy"]

    socres_3 = [val for val in results["kernel_size_3x3"].values()]
    scores_5 = [val for val in results["kernel_size_5x5"].values()]
    scores_7 = [val for val in results["kernel_size_7x7"].values()]
    scores_12 = [val for val in results["kernel_size_12x12"].values()]

    data = np.array([socres_3, scores_5, scores_7, scores_12])
    res = sp.posthoc_nemenyi_friedman(data.T)
    results_file.writelines("Porówanie dokładności sieci z różnymi rozmiarami splotu\n")
    results_file.writelines("Wyniki testu Friedmana oraz analizy post-hoc z użyciem testu Nemenyi:\n")
    results_file.writelines(str(res) + '\n\n\n')
