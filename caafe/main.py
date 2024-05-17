from preprocessing import make_datasets_numeric
# Automated Feature Engineering for tabular datasets
from sklearn_wrapper import CAAFEClassifier
# Fast Automated Machine Learning method for small tabular datasets
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier

import os
import tabpfn

import torch
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial
from data import (load_all_data,
                  get_data_split,
                  get_X_y)

import sys
from client import get_time, print_important

default_repeats = 2


def process(id, repeats=default_repeats):
    ds = cc_test_datasets_multiclass[id]
    ds, df_train, df_test, _, _ = get_data_split(ds, seed=0)
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    df_train, df_test = make_datasets_numeric(
        df_train, df_test, target_column_name)
    train_x, train_y = get_X_y(df_train, target_column_name)
    # test_x, test_y = get_X_y(df_test, target_column_name)

    # Setup Base Classifier

    # clf_no_feat_eng = RandomForestClassifier()
    clf_no_feat_eng = TabPFNClassifier(device=(
        'cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    clf_no_feat_eng.fit(train_x, train_y)
    # pred = clf_no_feat_eng.predict(test_x)
    # acc = accuracy_score(pred, test_y)

    # pred = clf_no_feat_eng.predict_proba(test_x)
    # acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, pred)
    # roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, pred)

    # Setup and Run CAAFE
    for i in range(repeats):
        print_important(
            f"start process {id} at {get_time()} with repeats {i}")
        caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                    llm_model="gpt-4",
                                    iterations=15)

        caafe_clf.fit_pandas(df_train,
                             target_column_name=target_column_name,
                             dataset_description=dataset_description)

    print_important(f"finish process {id} at {get_time()}")


def safe_process(id, repeats=default_repeats):
    try:
        process(id, repeats)
    except Exception as e:
        print_important(
            f"error in process {id} at {get_time()} with error {e}")


# 读取命令行参数
run_all = bool(sys.argv[1])
iteration_time = int(sys.argv[2])
id = int(sys.argv[3])

print_important(
    f"start running main.py with run_all: {run_all}, iteration_time: {iteration_time}, id: {id} at {get_time()}")

metric_used = tabular_metrics.auc_metric
cc_test_datasets_multiclass = load_all_data()
print_important(
    f"################################### LOAD IN {len(cc_test_datasets_multiclass)} DATASET ############################################")

if run_all:
    for i in range(0, iteration_time):
        if i == 0:
            for idx in range(id, len(cc_test_datasets_multiclass)):
                safe_process(idx)
        else:
            for idx in range(len(cc_test_datasets_multiclass)):
                safe_process(idx)

        print_important(f"finish pass {i} at {get_time()}")
else:
    process(id)
