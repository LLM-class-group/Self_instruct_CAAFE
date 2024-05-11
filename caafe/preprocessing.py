import pandas as pd
import copy
import numpy as np
from typing import Dict, Optional, Tuple


def create_mappings(df_train: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """
    为给定DataFrame中的分类列创建映射字典。
    
    参数:
        df_train (pd.DataFrame): 需要创建映射的DataFrame。
    
    返回:
        Dict[str, Dict[int, str]]: 分类列的映射字典。
    """
    mappings = {}
    for col in df_train.columns:
        if (
            df_train[col].dtype.name == "category"
            or df_train[col].dtype.name == "object"
        ):
            mappings[col] = {v: i for i, v in 
                enumerate(df_train[col].astype("category").cat.categories)
            }
    return mappings


def convert_categorical_to_integer_f(column: pd.Series, mapping: Optional[Dict[int, str]] = None) -> pd.Series:
    """
    使用给定的映射将分类列转换为整数值。
    
    参数:
        column (pd.Series): 要转换的列。
        mapping (Dict[int, str], 可选): 用于转换的映射。如果没有提供，则不进行转换。
    
    返回:
        pd.Series: 转换后的列。
    """
    if mapping is not None:
        # if column is categorical
        if column.dtype.name == "category":
            column = column.cat.add_categories([-1])
        return column.map(mapping).fillna(-1).astype(int)
    return column


def split_target_column(df: pd.DataFrame, target: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    将给定的DataFrame拆分为特征DataFrame和目标列。
    
    参数:
        df (pd.DataFrame): 要拆分的DataFrame。
        target (str, 可选): 目标列的名称。如果没有提供, 则返回None。
    
    返回:
        Tuple[pd.DataFrame, Optional[pd.Series]]: 特征DataFrame和目标列(如果存在)。
    """
    return (
        df[[c for c in df.columns if c != target]],
        df[target].astype(int) if (target and target in df.columns) else None,
    )


def make_dataset_numeric(df: pd.DataFrame, mappings: Dict[str, Dict[int, str]]) -> pd.DataFrame:
    """
    使用给定的映射将DataFrame中的分类列转换为整数值。
    
    参数:
        df (pd.DataFrame): 要转换的DataFrame。
        mappings (Dict[str, Dict[int, str]]): 用于转换的映射。
    
    返回:
        pd.DataFrame: 转换后的DataFrame。
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(
        lambda col: convert_categorical_to_integer_f(
            col, mapping=mappings.get(col.name)
        ),
        axis=0,
    )
    df = df.astype(float)

    return df


def make_datasets_numeric(df_train: pd.DataFrame, df_test: Optional[pd.DataFrame], target_column: str, return_mappings: Optional[bool] = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]:
    """
    使用从训练DataFrame创建的映射将训练和测试DataFrame中的分类列转换为整数值。
    
    参数:
        df_train (pd.DataFrame): 训练DataFrame。
        df_test (pd.DataFrame, 可选): 测试DataFrame。
        target_column (str): 目标列的名称。
        return_mappings (bool, 可选): 是否返回用于转换的映射。
    
    返回:
        Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]: 
        转换后的训练DataFrame、测试DataFrame(如果存在)，以及用于转换的映射(如果`return_mappings`为True)。
    """
    df_train = copy.deepcopy(df_train)
    df_train = df_train.infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test)
        df_test = df_test.infer_objects()

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train)

    # Apply the mappings to the train and test datasets
    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)

    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)

    if return_mappings:
        return df_train, df_test, mappings

    return df_train, df_test
