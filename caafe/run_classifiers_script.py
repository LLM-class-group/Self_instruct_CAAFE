"""
Runs downstream classifiers on the dataset and combines it with different feature sets.
"""

import argparse
import torch
from tabpfn.scripts import tabular_metrics
from data import load_all_data
from evaluate import evaluate_dataset_with_and_without_cafe

import os
from tabpfn import TabPFNClassifier
from functools import partial


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=-1,
    )
    args = parser.parse_args()
    tabpfn = TabPFNClassifier(
        N_ensemble_configurations=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tabpfn.fit = partial(
        tabpfn.fit, overwrite_warning=True
    )

    dataset_id = args.dataset_id
    seed = args.seed
    methods = ["random_forest", "xgb", "autogluon"]

    cc_test_datasets_multiclass = load_all_data()

    # if dataset_id is specified, only run on that dataset; if = -1, run on all datasets
    if dataset_id != -1:
        cc_test_datasets_multiclass = [cc_test_datasets_multiclass[dataset_id]]

    # set evaluation metric as AUC
    metric_used = tabular_metrics.auc_metric

    acc_sum = [0] * len(methods)
    auc_sum = [0] * len(methods)

    for i in range(0, len(cc_test_datasets_multiclass)):
        ds = cc_test_datasets_multiclass[i]
        acc_results_of_methods, auc_results_of_methods = evaluate_dataset_with_and_without_cafe(
            ds, seed, methods, metric_used
        )
        for j in range(0, len(methods)):
            acc_sum[j] += acc_results_of_methods[j]
            auc_sum[j] += auc_results_of_methods[j]

    acc_mean = [x / len(cc_test_datasets_multiclass) for x in acc_sum]
    auc_mean = [x / len(cc_test_datasets_multiclass) for x in auc_sum]

    with open("/home/jiahe/ML/Self_instruct_CAAFE/data/eval_result.txt", 'a') as eval_file:
        for i in range(0, len(methods)):
            if i == 0:
                method = "TabPFN"
            else:
                method = methods[i]
            eval_file.write(f"Method: {method}, AUC: {auc_mean[i]}, Accuracy: {acc_mean[i]}\n")
