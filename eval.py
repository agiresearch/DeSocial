import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data

from utils.metrics import get_link_prediction_metrics
from utils.utils import NegativeEdgeSampler
from utils.dataloader import Data

def evaluate(model_name: str, model: nn.Module, evaluate_idx_data_loader: DataLoader,
             evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
             data: Data = None, device = "cpu", is_test = False, neg_size = 4):
    """
        Evaluate models on the link prediction task
        Input:
            model_name: str, name of the model
            model: nn.Module, the model to be evaluated
            evaluate_idx_data_loader: DataLoader, evaluate index data loader
            evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
            evaluate_data: Data, data to be evaluated
            loss_func: nn.Module, loss function
            data: Data, data to be evaluated
            device: str, device to be used
            is_test: bool, whether is testing
            neg_size: int, number of negative samples

        Return:
            evaluate_losses: list, evaluate losses
            evaluate_metrics: list, evaluate metrics
            predicts: torch.Tensor, predicted probabilities
            evaluate_votes: list, evaluate Acc@2
            evaluate_ranks2: list, evaluate Acc@3
            evaluate_ranks3: list, evaluate Acc@5
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics, evaluate_votes, evaluate_ranks2, evaluate_ranks3 = [], [], [], [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random' and evaluate_neg_edge_sampler.negative_sample_strategy != 'random_with_collision_check':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                batch_neg_src_node_ids = []
                batch_neg_dst_node_ids = []
                for i in range(neg_size):
                    _, batch_neg_dst_node_ids_i = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids_i = batch_src_node_ids
                    batch_neg_src_node_ids.append(batch_neg_src_node_ids_i)
                    batch_neg_dst_node_ids.append(batch_neg_dst_node_ids_i)

                batch_neg_src_node_ids = np.concatenate(batch_neg_src_node_ids)
                batch_neg_dst_node_ids = np.concatenate(batch_neg_dst_node_ids)

            # get positive and negative probabilities, shape (batch_size, )
            batch_src_node_ids = torch.tensor(batch_src_node_ids).to(device)
            batch_dst_node_ids = torch.tensor(batch_dst_node_ids).to(device)
            batch_neg_src_node_ids = torch.tensor(batch_neg_src_node_ids).to(device)
            batch_neg_dst_node_ids = torch.tensor(batch_neg_dst_node_ids).to(device)

            edge_label_index = torch.cat([torch.stack([batch_src_node_ids, batch_dst_node_ids], dim=0), torch.stack([batch_neg_src_node_ids, batch_neg_dst_node_ids], dim=0)], dim=1).to(device)
            labels = torch.cat([torch.ones(batch_src_node_ids.size(0)), torch.zeros(batch_neg_src_node_ids.size(0))]).to(device)

            predicts = model(data, edge_label_index).squeeze(dim=-1).sigmoid()

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())
            eval = get_link_prediction_metrics(predicts=predicts, labels=labels, neg_size=neg_size)
            
            evaluate_metrics.append(eval[0])
            evaluate_votes.append(eval[1])
            evaluate_ranks2.append(eval[2])
            evaluate_ranks3.append(eval[3])

    return evaluate_losses, evaluate_metrics, predicts, evaluate_votes, evaluate_ranks2, evaluate_ranks3
