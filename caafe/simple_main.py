from preprocessing import make_datasets_numeric
# Automated Feature Engineering for tabular datasets
from sklearn_wrapper import CAAFEClassifier
# Fast Automated Machine Learning method for small tabular datasets
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier as MyClassifier

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

    clf_no_feat_eng = TabPFNClassifier(device=(
        'cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
    clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

    clf_no_feat_eng.fit(train_x, train_y)


    # Setup and Run CAAFE
    print_important(f"[{id}] dataset start at {get_time()}")
    for i in range(repeats):
        print_important(f"####### start cycle {i} at {get_time()} ########")
        caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                                    llm_model="gpt-4",
                                    iterations=10)
        caafe_clf.fit_pandas(df_train,
                             target_column_name=target_column_name,
                             dataset_description=dataset_description)

# 读取命令行参数

id = int(sys.argv[1])



metric_used = tabular_metrics.auc_metric
cc_test_datasets_multiclass = load_all_data()
print_important(
    f"################################### LOAD IN {len(cc_test_datasets_multiclass)} DATASET ############################################")    
process(id)
