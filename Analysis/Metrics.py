import numpy as np
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             log_loss,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             fbeta_score,
                             r2_score, 
                             explained_varience_score,
                             max_error,
                             median_absolute_error, 
                             mean_absolute_error,
                             mean_squared_error, 
                             mean_squared_log_error)

def mape(y_true, y_pred):
    """Mean Absolute Percent Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Classification(y_true, y_score, weights = None):
    
    scores = dict()

    if isinstance(y_score[0], int):
        scores['Accuracy'] = accuracy_score(y_true, y_score, sample_weight = weights)
        scores['ROC_AUC'] = roc_auc_score(y_true, y_score, sample_weight = weights)
    else:    
        scores['log_loss'] = log_loss(y_true, y_score, sample_weight = weights)

    scores['F1'] = f1_score(y_true, y_score, sample_weight = weights)
    scores['Precision'] = precision_score(y_true, y_score, sample_weight = weights)
    scores['Recall'] = precision_score(y_true, y_score, sample_weight = weights)
    scores['FBeta']  = fbeta_score(y_score, y_pred, sample_weight = weights)

    return scores

def Regression(y_true, y_pred, sample_weight = None):

    scores = dict()

    scores['Explained Varience'] = explained_varience_score(y_true, y_pred, sample_weight = sample_weight)
    scores['R2'] = r2_score(y_true, y_pred, sample_weight = sample_weight)
    scores['Max Error'] = max_error(y_true, y_pred)
    scores['MAPE'] = mape(y_true, y_pred)
    scores['MAE']  = mean_absolute_error(y_true, y_pred, sample_weight = sample_weight
    scores['MSE']  = mean_squared_error(y_true, y_pred, sample_weight = sample_weight)
    scores['Median Absolute Error'] = median_absolute_error(y_true, y_pred, sample_weight = sample_weight)

    return scores

