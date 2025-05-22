import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
import itertools
import random

def get_rank(target_score, candidate_score):
    """
        Get the rank of the target score among the candidate scores.
        Input:
            target_score: float, the target score
            candidate_score: list, the candidate scores
        Return:
            rank: int, the rank of the target score among the candidate scores
    """
    tmp_list = target_score - candidate_score
    rank = len(tmp_list[tmp_list <= 0]) + 1
    return rank

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor, neg_size: int = 4):
    """
        Get metrics for the link prediction task.
        Input:
            predicts: Tensor, shape (num_samples, )
            labels: Tensor, shape (num_samples, )
            neg_size: int, the number of negative samples
        Return:
            metrics: dictionary of metrics {'average_precision': ..., 'roc_auc': ..., 'accuracy': ...}
            acc_2: float, accuracy at 2 for each edge
            acc_3: float, accuracy at 3 for each edge
            acc_5: float, accuracy at 5 for each edge
    """
    predicts_np = predicts.cpu().detach().numpy()
    labels_np = labels.cpu().numpy()
    
    average_precision = average_precision_score(y_true=labels_np, y_score=predicts_np)
    roc_auc = roc_auc_score(y_true=labels_np, y_score=predicts_np)
    
    # Binarize predictions with threshold 0.5
    N = predicts.shape[0] // (neg_size + 1)

    pos_scores = torch.tensor(predicts_np[:N]).to(predicts.device)
    neg_scores = torch.tensor(predicts_np[N:N*2]).to(predicts.device).reshape(1, N)
    acc_2, acc_2_metrics = get_retrival_metrics(pos_scores, neg_scores)
    acc_2_metrics = {'Acc@2': acc_2_metrics}

    pos_scores = torch.tensor(predicts_np[:N]).to(predicts.device)
    neg_scores = torch.tensor(predicts_np[N:N*3]).to(predicts.device).reshape(2, N)
    acc_3, acc_3_metrics = get_retrival_metrics(pos_scores, neg_scores)
    acc_3_metrics = {'Acc@3': acc_3_metrics}

    pos_scores = torch.tensor(predicts_np[:N]).to(predicts.device)
    neg_scores = torch.tensor(predicts_np[N:]).to(predicts.device).reshape(4, N)
    acc_5, acc_5_metrics = get_retrival_metrics(pos_scores, neg_scores)
    acc_5_metrics = {'Acc@5': acc_5_metrics}

    return {
        'average_precision': average_precision,
        'roc_auc': roc_auc,
        'weight': predicts.shape[0],
        **acc_2_metrics,
        **acc_3_metrics,
        **acc_5_metrics
    }, acc_2, acc_3, acc_5

def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
        Get metrics for the link prediction task
        Input:
            pos_scores: Tensor, shape (num_samples, )
            neg_scores: Tensor, shape (neg_size, num_samples)
        Return:
            H1: whether the ground truth ranked the first.
            H1 mean: the average of H1.
    """
    try:
        pos_scores = pos_scores.cpu().detach().numpy()
    except:
        pass
    try:
        neg_scores = np.array([sub_score.cpu().numpy() for sub_score in neg_scores]).T # num_samples * neg_size
    except:
        neg_scores = np.array([sub_score for sub_score in neg_scores]).T # num_samples * neg_size

    H1 = []
    ranks = []
    for i in range(len(pos_scores)):
        rank = get_rank(pos_scores[i], neg_scores[i])
        if rank <= 1:
            H1.append(1)
        else:
            H1.append(0)

    return H1, np.mean(H1)
