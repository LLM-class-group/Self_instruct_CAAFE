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
                  get_X_y
                  )

import sys


def process(id):
    ds = cc_test_datasets_multiclass[id]
    ds, df_train, df_test, _, _ = get_data_split(ds, seed=0)
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]
    df_train, df_test = make_datasets_numeric(
        df_train, df_test, target_column_name)
    train_x, train_y = get_X_y(df_train, target_column_name)
    test_x, test_y = get_X_y(df_test, target_column_name)

    # Setup Base Classifier

    # clf_no_feat_eng = RandomForestClassifier()
    clf_no_feat_eng = TabPFNClassifier(device=(
        'cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    clf_no_feat_eng.fit(train_x, train_y)
    # pred = clf_no_feat_eng.predict(test_x)
    # acc = accuracy_score(pred, test_y)

    pred = clf_no_feat_eng.predict_proba(test_x)
    acc = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, pred)
    roc = tabpfn.scripts.tabular_metrics.auc_metric(test_y, pred)

    # Setup and Run CAAFE

    caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                llm_model="gpt-4",
                                iterations=100)

    caafe_clf.fit_pandas(df_train,
                         target_column_name=target_column_name,
                         dataset_description=dataset_description)

    # pred_p_test = caafe_clf.predict_proba(df_test)
    # pred_p_train = caafe_clf.predict_proba(df_train)

    # acc_test = tabpfn.scripts.tabular_metrics.accuracy_metric(test_y, pred_p_test)
    # acc_train = tabpfn.scripts.tabular_metrics.accuracy_metric(train_y, pred_p_train)
    # roc_test = tabpfn.scripts.tabular_metrics.auc_metric(test_y, pred_p_test)
    # pred_test = caafe_clf.predict(df_test)
    # pred_train = caafe_clf.predict(df_train)

    # acc_test = accuracy_score(pred_test, test_y)
    # # acc_train = accuracy_score(pred_train, train_y)

    # print(pred_p_test)
    # print(pred)
    # print(caafe_clf.code)

    # from datetime import datetime
    # # 获取当前时间
    # current_time = datetime.now()
    # # 将关键信息输出到 main_output.txt
    # output_path = "/home/jiahe/ML/Self_instruct_CAAFE/output/main_output.txt"
    # with open(output_path, 'w') as f:
    #     print(f"Processing CAAFE in dataset {id}", file=f)
    #     print(f"Test accuracy before CAAFE {acc}", file=f)
    #     print(f"Test accuracy after CAAFE {acc_test}", file=f)
    #     print(f"Test ROC before CAAFE {roc}", file=f)
    #     print(f"Test ROC after CAAFE {roc_test}", file=f)
    #     print(f"Current time: {current_time}", file=f)

    # print(f'Train accuracy after CAAFE {acc_train}')


# color
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'  # reset


# 读取命令行参数
run_all = bool(sys.argv[1])
id = int(sys.argv[2])

metric_used = tabular_metrics.auc_metric
cc_test_datasets_multiclass = load_all_data()
print(
    f"################################### LOAD IN {len(cc_test_datasets_multiclass)} DATASET ############################################")

if run_all:
    for id in range(len(cc_test_datasets_multiclass)):
        process(id)
else:
    process(id)


