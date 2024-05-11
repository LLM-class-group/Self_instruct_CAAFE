from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import torch
import numpy as np


def auc_metric(target, pred, multi_class="ovo", numpy=False):
    """
    计算AUC分数, 支持二分类和多分类问题。
    
    参数:
        target (array): 真实的标签。
        pred (array): 模型预测的输出（概率或得分）。
        multi_class (str): 多分类问题的处理方式，"ovo"表示一对一，"ovr"表示一对全。
        numpy (bool): 指示是否使用NumPy(True)或PyTorch(False)进行计算。
        
    返回:
        AUC分数,如果计算过程中发生错误,则返回NaN。
    """
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
        if len(lib.unique(target)) > 2:
            if not numpy:
                return torch.tensor(
                    roc_auc_score(target, pred, multi_class=multi_class)
                )
            return roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            if not numpy:
                return torch.tensor(roc_auc_score(target, pred))
            return roc_auc_score(target, pred)
    except ValueError as e:
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)


def accuracy_metric(target, pred):
    """
    计算模型的准确率。
    
    参数:
        target (array): 真实的标签。
        pred (array): 模型预测的输出（类别或概率）。
        
    返回:
        准确率。
    """
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))
