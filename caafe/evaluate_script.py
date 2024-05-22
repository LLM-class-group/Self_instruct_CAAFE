"""
Runs the CAAFE algorithm on a dataset and saves the generated code and prompt to a file.
Then, runs classifiers on the dataset with or without the generated features.
"""

import argparse
from functools import partial

from tabpfn.scripts import tabular_metrics
from tabpfn import TabPFNClassifier
import os
import torch
import warnings
import numpy as np


from data import get_data_split, load_all_data
from caafe import generate_features
from evaluate import evaluate_dataset_with_or_without_caafe

sft = True
generate = True
dirname = "generated_code_sft2" if sft else "generated_code"


def generate_and_save_feats(i, seed=0, iterative_method=None, iterations=10):
    if iterative_method is None:
        iterative_method = tabpfn  # default method

    ds = cc_test_datasets_multiclass[i]

    # we also need to reformat ds to df
    ds, df_train, df_test, df_train_old, df_test_old = get_data_split(ds, seed)
    code, prompt, messages = generate_features(
        ds,
        df_train,
        just_print_prompt=False,
        model=model,
        iterative=iterations,
        metric_used=metric_used,
        iterative_method=iterative_method,
        display_method="print",
    )

    data_dir = os.environ.get("DATA_DIR", "data")
    f = open(
        f"{data_dir}/{dirname}/{ds[0]}_{seed}_prompt.txt",
        "w",
    )
    f.write(prompt)
    f.close()

    f = open(
        f"{data_dir}/{dirname}/{ds[0]}_{seed}_code.txt", "w")
    f.write(code)
    f.close()


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
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--using_caafe",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    dataset_id = args.dataset_id
    iterations = args.iterations
    seed = args.seed
    using_caafe = args.using_caafe

    model = ""

    cc_test_datasets_multiclass = load_all_data()

    # if dataset_id is specified, only run on that dataset; if = -1, run on all datasets
    if dataset_id != -1:
        cc_test_datasets_multiclass = [cc_test_datasets_multiclass[dataset_id]]

    # set evaluation metric as AUC
    metric_used = tabular_metrics.auc_metric

    tabpfn = TabPFNClassifier(
        N_ensemble_configurations=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    tabpfn.fit = partial(
        tabpfn.fit, overwrite_warning=True
    )

    # generate and save code
    if generate:
        for i in range(0, len(cc_test_datasets_multiclass)):
            generate_and_save_feats(i, seed=seed, iterations=iterations)

    methods = [tabpfn, "random_forest", "xgb"]

    acc = [[0 for _ in range(len(cc_test_datasets_multiclass))]
           for _ in range(len(methods))]
    auc = [[0 for _ in range(len(cc_test_datasets_multiclass))]
           for _ in range(len(methods))]

    acc_sum = [0] * len(methods)
    auc_sum = [0] * len(methods)

    for i in range(0, len(cc_test_datasets_multiclass)):
        ds = cc_test_datasets_multiclass[i]
        acc_results_of_methods, auc_results_of_methods = evaluate_dataset_with_or_without_caafe(
            ds, seed, methods, metric_used, using_caafe
        )
        for j in range(0, len(methods)):
            acc_sum[j] += acc_results_of_methods[j]
            auc_sum[j] += auc_results_of_methods[j]
            acc[j][i] = acc_results_of_methods[j]
            auc[j][i] = auc_results_of_methods[j]

    acc_mean = [x / len(cc_test_datasets_multiclass) for x in acc_sum]
    auc_mean = [x / len(cc_test_datasets_multiclass) for x in auc_sum]

    with open("/home/jiahe/ML/Self_instruct_CAAFE/data/eval_result.txt", 'a') as eval_file:
        if using_caafe:
            eval_file.write(f"Using CAAFE:\n")
        else:
            eval_file.write(f"No feature engineering:\n")
        for i in range(0, len(methods)):
            if i == 0:
                method = "TabPFN"
            else:
                method = methods[i]
            eval_file.write(
                f"Method: {method}, Average: AUC = {auc_mean[i]}, Accuracy = {acc_mean[i]}\n")
            for j in range(0, len(cc_test_datasets_multiclass)):
                eval_file.write(
                    f"Dataset {j}: AUC = {auc[i][j]}, Accuracy = {acc[i][j]}\n")
            eval_file.write("\n")
