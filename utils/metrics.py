import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import average_precision_score, roc_auc_score
import itertools
import random

def get_rank(target_score, candidate_score):
    tmp_list = target_score - candidate_score
    rank = len(tmp_list[tmp_list <= 0]) + 1
    return rank

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor, neg_size: int = 4):
    """
    Get metrics for the link prediction task.
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'average_precision': ..., 'roc_auc': ..., 'accuracy': ...}
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

def get_vote_avg_metrics(test_votes_all: list, test_ranks2_all: list, test_ranks3_all: list, test_acc: list, test_ranks2_single: list, test_ranks3_single: list, exp: int = 7, sample_groups: int = 3):
    experts_all = len(test_votes_all)
    assert experts_all >= exp, "Not enough experts to sample from."

    all_combs = list(itertools.combinations(range(experts_all), exp))
    selected_combs = random.sample(all_combs, k=sample_groups)
    vote_acc_list = []
    avg_acc_list = []
    vote_rank2_list = []
    avg_rank2_list = []
    vote_rank3_list = []
    avg_rank3_list = []

    for comb in selected_combs:
        votes = torch.stack([test_votes_all[i] for i in comb], dim=0)
        vote_sum = torch.sum(votes, dim=0)
        final_votes = (vote_sum > (exp // 2)).float()
        vote_acc = torch.mean(final_votes).item()
        avg_acc = np.mean([test_acc[i] for i in comb])
        vote_acc_list.append(vote_acc)
        avg_acc_list.append(avg_acc)
    
    for comb in selected_combs:
        ranks2 = torch.stack([test_ranks2_all[i] for i in comb], dim=0)
        rank2_sum = torch.sum(ranks2, dim=0)
        final_votes2 = (rank2_sum > (exp // 2)).float()
        vote_rank2_acc = torch.mean(final_votes2).item()
        avg_rank2 = np.mean([test_ranks2_single[i] for i in comb])
        vote_rank2_list.append(vote_rank2_acc)
        avg_rank2_list.append(avg_rank2)

    for comb in selected_combs:
        ranks3 = torch.stack([test_ranks3_all[i] for i in comb], dim=0)
        rank3_sum = torch.sum(ranks3, dim=0)
        final_votes3 = (rank3_sum > (exp // 2)).float()
        vote_rank3_acc = torch.mean(final_votes3).item()
        avg_rank3 = np.mean([test_ranks3_single[i] for i in comb])
        vote_rank3_list.append(vote_rank3_acc)
        avg_rank3_list.append(avg_rank3)

    vote_acc_mean = np.round(np.mean(vote_acc_list), 4)
    vote_acc_std = np.round(np.std(vote_acc_list), 4)
    avg_acc_mean = np.round(np.mean(avg_acc_list), 4)
    avg_acc_std = np.round(np.std(avg_acc_list), 4)
    vote_rank2_acc_mean = np.round(np.mean(vote_rank2_list), 4)
    vote_rank2_acc_std = np.round(np.std(vote_rank2_list), 4)
    avg_rank2_mean = np.round(np.mean(avg_rank2_list), 4)
    avg_rank2_std = np.round(np.std(avg_rank2_list), 4)
    vote_rank3_acc_mean = np.round(np.mean(vote_rank3_list), 4)
    vote_rank3_acc_std = np.round(np.std(vote_rank3_list), 4)
    avg_rank3_mean = np.round(np.mean(avg_rank3_list), 4)
    avg_rank3_std = np.round(np.std(avg_rank3_list), 4)

    return avg_acc_mean, avg_acc_std, vote_acc_mean, vote_acc_std, avg_rank2_mean, avg_rank2_std, vote_rank2_acc_mean, vote_rank2_acc_std, avg_rank3_mean, avg_rank3_std, vote_rank3_acc_mean, vote_rank3_acc_std

def get_retrival_metrics(pos_scores: torch.Tensor, neg_scores: torch.Tensor):
    """
    get metrics for the link prediction task
    :param pos_scores: Tensor, shape (num_samples, )
    :param neg_scores: Tensor, shape (neg_size, num_samples)
    :return:
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
