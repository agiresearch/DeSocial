import random
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data

class Selection():
    def __init__(self, train_data, val_data, current_time, num_nodes, node_raw_features, device, batch_size=32768, alpha=0):
        super(Selection, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.current_time = current_time
        self.num_nodes = num_nodes
        self.node_raw_features = node_raw_features
        self.backbones = []
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha

    def rule_based_selection(self, degree, clustering):
        if degree >= 6:
            return "SGC"
        if clustering < 0.2 and degree >= 4:
            return "SAGE"
        if degree <= 2:
            return "MLP"
        if clustering >= 0.4:
            return "GCN"
        return "GAT"

    def extract_features(self, node_id, neighbor_dict, degree_dict):
        neighbors = neighbor_dict.get(node_id, set())
        degree = len(neighbors)

        if degree == 0:
            return [0, 0, 0, 0, 0, 0]

        neighbor_degrees = [degree_dict.get(n, 0) for n in neighbors]
        avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0.0
        neighbor_var = np.var(neighbor_degrees) if neighbor_degrees else 0.0

        links = 0
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2 and n2 in neighbor_dict.get(n1, set()):
                    links += 1
        links = links / 2

        clustering = links / (degree * (degree - 1) / 2) if degree > 1 else 0.0
        ego_edge_count = links

        second_neighbors = set()
        for nbr in neighbors:
            second_neighbors.update(neighbor_dict.get(nbr, set()))
        second_neighbors.discard(node_id)
        second_neighbors = second_neighbors - neighbors
        second_hop_count = len(second_neighbors)

        return [degree, avg_neighbor_degree, clustering, ego_edge_count, neighbor_var, second_hop_count]

    def backbone_selection(self, 
                           num_neighbor_samples=30, 
                           selection_mechanism="auto"):
        src_node_ids = np.concatenate([self.train_data.src_node_ids, self.val_data.src_node_ids])
        dst_node_ids = np.concatenate([self.train_data.dst_node_ids, self.val_data.dst_node_ids])
        node_interact_times = np.concatenate([self.train_data.node_interact_times, self.val_data.node_interact_times])
        #alpha = -0.01
        #t_max = node_interact_times.max()
        #node_interact_time_weights = np.exp(alpha * (t_max - node_interact_times))
        #node_interact_time_weights = torch.tensor(node_interact_time_weights, dtype=torch.float, device=self.device)
        #node_interact_time_weights = node_interact_time_weights / node_interact_time_weights.sum()

        # Preprocess neighbors
        neighbor_dict = defaultdict(set)
        for u, v, t in zip(src_node_ids, dst_node_ids, node_interact_times):
            neighbor_dict[u].add((v, t))
            neighbor_dict[v].add((u, t))

        # Preprocess degree
        degree_dict = {node: len(neighbors) for node, neighbors in neighbor_dict.items()}
        selected_backbones = np.zeros((self.num_nodes,))

        if selection_mechanism == "auto":
            # BCDSN adopted this selection mechanism
            edge_label_index = []
            edge_label_index_neg = []
            node_interact_time_weights = []
            neg_dst = np.random.randint(0, self.num_nodes, self.num_nodes * num_neighbor_samples)
            for node_id in tqdm(range(self.num_nodes)):
                neighbors_all = list(neighbor_dict[node_id])
                if len(neighbors_all) == 0:
                    # just for padding
                    neighbors_all = [(0, 0)]

                # select neighbor list for node_id
                neighbors = [random.choice(neighbors_all) for _ in range(num_neighbor_samples)]
                #replace_flag = num_neighbor_samples > len(neighbors_all)
                #neighbors = np.random.choice(neighbors_all, size=num_neighbor_samples, replace=replace_flag)
                for i in range(len(neighbors)):
                    edge_label_index.append([node_id, neighbors[i][0]])
                    edge_label_index_neg.append([node_id, neg_dst[i + node_id * num_neighbor_samples]])
                    node_interact_time_weights.append(np.exp(self.alpha * (self.current_time - neighbors[i][1])))
                    #node_interact_time_weights.append(1)

            edge_label_index.extend(edge_label_index_neg)
            
            # Here can process by batches.
            edge_label_index = torch.tensor(np.array(edge_label_index).T, dtype=torch.long, device=self.device)
            x = torch.tensor(self.node_raw_features, dtype=torch.float, device=self.device)
            edge_index_dir = torch.tensor([src_node_ids, dst_node_ids], dtype=torch.long)
            edge_index_inv = torch.tensor([dst_node_ids, src_node_ids], dtype=torch.long)
            edge_index = torch.cat([edge_index_dir, edge_index_inv], dim=1)

            node_interact_time_weights = np.array(node_interact_time_weights)

            data = Data(x=x, edge_index=edge_index)

            backbone_node_prob = np.zeros((self.num_nodes, len(self.backbones)))

            for backbone_id in range(len(self.backbones)):
                # use each backbone to do the prediction
                model = self.backbones[backbone_id].to(self.device)
                # node_raw_feature is the input, predicted probabilities is the output.
                all_preds = []
                num_edges = edge_label_index.shape[1]
                for i in tqdm(range(0, num_edges, self.batch_size)):
                    batch_indices = edge_label_index[:, i:i+self.batch_size]
                    with torch.no_grad():
                        pred = model(data.to(self.device), batch_indices).squeeze(dim=-1).sigmoid()
                        all_preds.append(pred.cpu())

                pred = torch.cat(all_preds, dim=0)

                for node_id in range(self.num_nodes):
                    backbone_pred_pos = pred[node_id * num_neighbor_samples: (node_id + 1) * num_neighbor_samples].cpu().detach().numpy()
                    backbone_pred_neg = pred[(node_id + self.num_nodes) * num_neighbor_samples: (node_id + 1 + self.num_nodes) * num_neighbor_samples].cpu().detach().numpy()
                    backbone_pred = np.array(backbone_pred_pos > backbone_pred_neg, dtype=np.float64) * node_interact_time_weights[node_id * num_neighbor_samples: (node_id + 1) * num_neighbor_samples]
                    #backbone_pred = np.array(backbone_pred_pos > backbone_pred_neg, dtype=np.float64)
                    backbone_node_prob[node_id, backbone_id] = backbone_pred.mean()

            # select the backbone with the highest probability for each node
            selected_backbones = np.argmax(backbone_node_prob, axis=1)

        else:
            x_in = np.zeros((self.num_nodes, 6))
            for node_id in tqdm(range(self.num_nodes)):
                # construct input vector
                edge_features = self.extract_features(node_id, neighbor_dict, degree_dict)
                x_in[node_id] = edge_features
            for node_id in tqdm(range(self.num_nodes)):

                if selection_mechanism == "random":
                    selected_backbone = random.choice([0, 1, 2, 3, 4])
                
                elif selection_mechanism == "rule":
                    degree, clustering = x_in[node_id][0], x_in[node_id][2]
                    selected_backbone = self.rule_based_selection(degree, clustering)
                
                selected_backbones.append(selected_backbone)

        return selected_backbones
