from .f1_score_f1_pa import *
from .fc_score import *
from .precision_at_k import *
from .customizable_f1_score import *
from .AUC import *
from .Matthews_correlation_coefficient import *
from .affiliation.generics import convert_vector_to_events
from .affiliation.metrics import pr_from_events
from .vus.models.feature import Window
from .vus.metrics import get_range_vus_roc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, average_precision_score


def combine_all_evaluation_scores(gt, pred, anomaly_scores):
    """
    计算所有评价指标
    参数:
        gt: ground truth labels (真实标签)
        pred: predicted labels (预测标签)
        anomaly_scores: anomaly scores (异常分数)
    """
    events_pred = convert_vector_to_events(pred)
    events_gt = convert_vector_to_events(gt)
    Trange = (0, len(pred))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    P = affiliation['precision']
    R = affiliation['recall']
    affiliation_F = 2 * P * R / (P + R + 1e-8)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary',
                                                                          zero_division=0)
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(pred, gt)

    # 计算 AUC-ROC 和 AUC-PR
    try:
        auc_roc = roc_auc_score(gt, anomaly_scores)
    except:
        auc_roc = float('nan')
    try:
        auc_pr = average_precision_score(gt, anomaly_scores)
    except:
        auc_pr = float('nan')

    vus_results = get_range_vus_roc(pred, gt, 100)  # default slidingWindow = 100

    score_list_simple = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f_score": f_score,
                    "pa_accuracy": pa_accuracy,
                    "pa_precision": pa_precision,
                    "pa_recall": pa_recall,
                    "pa_f_score": pa_f_score,
                    "Affiliation precision": affiliation['precision'],
                    "Affiliation recall": affiliation['recall'],
                    "affiliation_F1": affiliation_F,
                    "R_AUC_ROC": vus_results["R_AUC_ROC"],
                    "R_AUC_PR": vus_results["R_AUC_PR"],
                    "AUC_ROC": auc_roc,
                    "AUC_PR": auc_pr,
                    "VUS_ROC": vus_results["VUS_ROC"],
                    "VUS_PR": vus_results["VUS_PR"]
                  }
    return score_list_simple


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 1
    pred_labels[55:62] = 1
    anomaly_scores = np.zeros(100)
    anomaly_scores[15:17] = 0.7
    anomaly_scores[55:62] = 0.6
    pred_labels[51:55] = 1
    scores = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)
    for key,value in scores.items():
        print(key,' : ',value)

    
if __name__ == "__main__":
    main()