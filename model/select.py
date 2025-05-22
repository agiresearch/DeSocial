import random
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data

class Selection():
    def __init__(self, train_data, val_data, current_time, num_nodes, request_nodes, node_raw_features, device, batch_size=32768, alpha=0, gamma = 250):
        super(Selection, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.current_time = current_time
        self.num_nodes = num_nodes
        self.request_nodes = request_nodes
        self.node_raw_features = node_raw_features
        self.backbones = []
        self.device = device
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma

    def rule_based_selection(self, degree, clustering):
        """
            Simple rule-based selection of backbone models based on node features.
        """

        if degree >= 6:
            if "SGC" in self.backbones:
                return "SGC"
        if clustering < 0.2 and degree >= 4:
            if "SAGE" in self.backbones:
                return "SAGE"
        if degree <= 2:
            if "MLP" in self.backbones:
                return "MLP"
        if clustering >= 0.4:
            if "GCN" in self.backbones:
                return "GCN"
        return self.backbones[-1]

    def extract_features(self, node_id, neighbor_dict, degree_dict):
        """
            Extract features for a given node based on its local subgraph.
        """

        neighbors = neighbor_dict.get(node_id, set())
        degree = len(neighbors)

        if degree == 0:
            return [0, 0]

        neighbor_degrees = [degree_dict.get(n, 0) for n in neighbors]
        avg_neighbor_degree = np.mean(neighbor_degrees)
        neighbor_var = np.var(neighbor_degrees)

        links = sum(
            1 for i, n1 in enumerate(neighbors)
            for n2 in list(neighbors)[i+1:]
            if n2 in neighbor_dict.get(n1, set())
        )
        clustering = links / (degree * (degree - 1) / 2) if degree > 1 else 0.0

        return [degree, clustering]

    def neighbor_setup(self):
        """
            Setup the neighbor information for all nodes.
        """
        self.src_node_ids = np.concatenate([self.train_data.src_node_ids, self.val_data.src_node_ids])
        self.dst_node_ids = np.concatenate([self.train_data.dst_node_ids, self.val_data.dst_node_ids])
        node_interact_times = np.concatenate([self.train_data.node_interact_times, self.val_data.node_interact_times])

        # Preprocess neighbors
        self.neighbor_dict = defaultdict(set)
        for u, v, t in zip(self.src_node_ids, self.dst_node_ids, node_interact_times):
            self.neighbor_dict[u].add((v, t))
            self.neighbor_dict[v].add((u, t))

        # Preprocess degree
        self.degree_dict = {node: len(neighbors) for node, neighbors in self.neighbor_dict.items()}
        self.selected_backbones = np.zeros((self.request_nodes,))
    
    def create_neighbor_samples(self, node_id):
        """
            Create positive and negative neighbor samples for a given node.
        """

        neg_dst = np.random.randint(0, self.num_nodes, self.gamma)
        edge_label_index = []
        edge_label_index_neg = []
        node_interact_time_weights = []
        neighbors_all = list(self.neighbor_dict[node_id])
        if len(neighbors_all) == 0:
            # just for padding
            neighbors_all = [(0, 0)]
        
        neighbors = [random.choice(neighbors_all) for _ in range(self.gamma)]
        for i in range(len(neighbors)):
            edge_label_index.append([node_id, neighbors[i][0]])
            edge_label_index_neg.append([node_id, neg_dst[i]])
            node_interact_time_weights.append(np.exp(self.alpha * (self.current_time - neighbors[i][1])))
        
        return edge_label_index, edge_label_index_neg, node_interact_time_weights
    
    def take_exam(self, pos_neighbor, neg_neighbor, neighbor_weights, backbone_id):
        """
            Perform the neighbor sample tasks for all the requesters, and return the test results.
            Input:
                pos_neighbor: Positive neighbor samples
                neg_neighbor: Negative neighbor samples
                backbone_id: The ID of the backbone model to use
            Return:
                backbone_node_prob: The test results for the selected backbone model
        """

        edge_label_index = pos_neighbor
        edge_label_index.extend(neg_neighbor)
        # Here can process by batches.
        edge_label_index = torch.tensor(np.array(edge_label_index).T, dtype=torch.long, device=self.device)
        x = torch.tensor(self.node_raw_features, dtype=torch.float, device=self.device)
        edge_index_dir = torch.tensor([self.src_node_ids, self.dst_node_ids], dtype=torch.long)
        edge_index_inv = torch.tensor([self.dst_node_ids, self.src_node_ids], dtype=torch.long)
        edge_index = torch.cat([edge_index_dir, edge_index_inv], dim=1)

        node_interact_time_weights = np.array(neighbor_weights)
        data = Data(x=x, edge_index=edge_index)
        
        backbone_node_prob = np.zeros(self.request_nodes, )

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

        for node_id in range(self.request_nodes):
            backbone_pred_pos = pred[node_id * self.gamma: (node_id + 1) * self.gamma].cpu().detach().numpy()
            backbone_pred_neg = pred[(node_id + self.request_nodes) * self.gamma: (node_id + 1 + self.request_nodes) * self.gamma].cpu().detach().numpy()
            backbone_pred = np.array(backbone_pred_pos > backbone_pred_neg, dtype=np.float64) * node_interact_time_weights[node_id * self.gamma: (node_id + 1) * self.gamma]
            backbone_node_prob[node_id] = backbone_pred.mean()

        return backbone_node_prob
