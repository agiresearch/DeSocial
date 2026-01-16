import numpy as np
from utils.dataloader import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.utils import set_random_seed, create_optimizer
from utils.utils import NegativeEdgeSampler
from utils.dataloader import Data as MyData
from utils.dataloader import get_idx_data_loader, get_link_prediction_data
from utils.earlystopping import EarlyStopping
from utils.configs import get_link_prediction_args, load_link_prediction_best_configs, load_lr_given_models, get_num_users
from model.dispatcher import Dispatcher
from eval import evaluate
import os
from tqdm import tqdm
import time

# Global lists, storing all the user objects.
user_storage = []

class User:
    """
    User class for the distributed social recommendation system.
    Each user maintains their own model and local data.
    """

    def __init__(self, user_id, num_of_val, num_nodes, coordinator=None):
        """
        Initialize a user.
        
        Parameters:
            user_id: The ID of the user.
            num_of_val: The number of validators in the system.
            num_nodes: The number of nodes in the system.
            coordinator: Reference to the coordinator (optional).
        """
        # System Properties
        self.coordinator = coordinator
        self.num_of_val = num_of_val
        
        # Local Data Storage
        self.user_id = user_id
        self.model = None  # Will initialize in run.py
        self.optimizer = None  # Will initialize in run.py
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.requests_collected = []
        self.num_nodes = num_nodes

        # Local Data Storage (User Backbone Selection)
        self.pa_requesters = []
        self.pos_neighbor = []
        self.neg_neighbor = []
        self.neighbor_weight = []
        self.task_result = []
        self.task_results_to_comp = []
        self.validation_results = {}
        self.neighbor_sample_list = []
    
    def send_a_request(self, target, inter_terminal, timestamp):
        """
        Send a request to predict a link with the target node.
        
        Parameters:
            target: The target node.
            inter_terminal: The intermediate terminal node.
            timestamp: The timestamp of the request.
        """
        # Request is recorded locally
        pass

    def retrieve_test_data(self, inter_terminal):
        """
        Retrieve test data from the intermediate terminal node.
        
        Parameters:
            inter_terminal: The intermediate terminal node.
        """
        self.test_data = user_storage[inter_terminal].test_data

    def select_validators(self, val_tot, val_num):
        """
        Request the coordinator to select validators randomly.
        
        Parameters:
            val_tot: The total number of validators in a specified backbone community.
            val_num: The number of validators to be chosen.
            
        Returns:
            list: The list of validator indices.
        """
        if self.coordinator:
            validators = self.coordinator.select_validators(val_tot, val_num)
        else:
            # Fallback to local random selection
            validators = list(np.random.choice(val_tot, min(val_num, val_tot), replace=False))
        return validators

    def vote(self, agree):
        """
        Submit a vote based on the model prediction.
        
        Parameters:
            agree (bool): The vote of the validator.
        """
        if self.coordinator:
            self.coordinator.collect_vote(self.user_id, agree)

    def update_social_network(self):
        """
        Update the social network data.
        Two steps:
            1. Merge the old train data and the old val data to new train data.
            2. The new val data is the test data.
        """
        src_node_ids = np.concatenate((self.train_data.src_node_ids, self.val_data.src_node_ids), axis=0)
        dst_node_ids = np.concatenate((self.train_data.dst_node_ids, self.val_data.dst_node_ids), axis=0)
        node_interact_times = np.concatenate((self.train_data.node_interact_times, self.val_data.node_interact_times), axis=0)
        self.train_data = MyData(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
        self.val_data = self.test_data
    
    def give_prediction(self, logger, args, save_model_folder, model_name_with_params):
        """
        Train the model and give predictions.
        
        Parameters:
            logger: The logger object.
            args: The arguments for the model.
            save_model_folder: The folder to save the model.
            model_name_with_params: The name of the model with parameters.
        """
        train_data = self.train_data
        val_data = self.val_data
        test_data = self.test_data

        edge_index = torch.tensor([train_data.src_node_ids, train_data.dst_node_ids], dtype=torch.long)
        edge_index_dir = edge_index
        edge_index_inv = torch.tensor([train_data.dst_node_ids, train_data.src_node_ids], dtype=torch.long)
        edge_index = torch.cat([edge_index_dir, edge_index_inv], dim=1)

        train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), 
                                                    batch_size=args.batch_size, 
                                                    shuffle=True)
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), 
                                                batch_size=args.batch_size, 
                                                shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), 
                                                batch_size=args.batch_size, 
                                                shuffle=False)

        x = torch.tensor(self.node_raw_features, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index).to(device=args.device)
        model = self.model.to(args.device)
        optimizer = self.optimizer
        set_random_seed(seed=self.user_id)
        loss_func = nn.BCELoss()

        train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, 
                                                    dst_node_ids=train_data.dst_node_ids,  
                                                    seed=self.user_id)
        val_neg_edge_sampler = NegativeEdgeSampler(num_nodes=self.num_nodes,
                                                   seed=self.user_id)
        test_neg_edge_sampler = NegativeEdgeSampler(num_nodes=self.num_nodes,
                                                    seed=self.user_id)
        
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder, save_model_name=model_name_with_params, logger=logger, model_name=args.model_name)

        for epoch in range(args.num_epochs):
            model.train()
            train_losses, train_metrics = [], []
            for batch_indices in tqdm(train_idx_data_loader, ncols=120):
                batch_indices = batch_indices.numpy()
                src = torch.tensor(train_data.src_node_ids[batch_indices], device=args.device)
                dst = torch.tensor(train_data.dst_node_ids[batch_indices], device=args.device)
                _, neg_dst = train_neg_edge_sampler.sample(len(src))
                neg_dst = torch.tensor(neg_dst, device=args.device)
                edge_label_index = torch.cat([torch.stack([src, dst], dim=0), torch.stack([src, neg_dst], dim=0)], dim=1)
                edge_label = torch.cat([torch.ones(src.size(0)), torch.zeros(src.size(0))]).to(args.device)
                pred = model(data.to(args.device), edge_label_index).squeeze(dim=-1).sigmoid()
                loss = loss_func(pred, edge_label)
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch + 1}: Train loss {np.mean(train_losses):.4f}")

            if (epoch + 1) % args.test_interval_epochs == 0:
                val_losses, val_metrics, _, _, _, _ = evaluate(model_name=args.model_name, model=model,
                    evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler,
                    evaluate_data=val_data, loss_func=loss_func,  data=data, device=args.device)

                val_metric_indicator = [('val loss', np.mean(val_losses), False)]
                logger.info(f"val loss: {np.mean(val_losses)}")
                early_stop = early_stopping.step(val_metric_indicator, model)
                if early_stop:
                    break

        early_stopping.load_checkpoint(model)
        logger.info("Training finished. Best model loaded.")

        test_losses, test_metrics, _, test_acc_2_vote, test_acc_3_vote, test_acc_5_vote = evaluate(model_name=args.model_name, model=model,
            evaluate_idx_data_loader=test_idx_data_loader, evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data, loss_func=loss_func, data=data, device=args.device, is_test = True)
        
        logger.info(f"Final test loss: {np.mean(test_losses):.4f}")
        weights = np.array([m["weight"] for m in test_metrics])
        for metric in test_metrics[0].keys():
            values = np.array([m[metric] for m in test_metrics])
            metric_value = np.sum(values * weights) / np.sum(weights)
            logger.info(f"Test {metric}: {metric_value:.4f}")
            if metric == "Acc@2":
                acc_2 = metric_value
            elif metric == "Acc@3":
                acc_3 = metric_value
            elif metric == "Acc@5":
                acc_5 = metric_value
        
        model = model.cpu()
        self.model = model
        self.optimizer = optimizer

        # Store the validation results locally
        logger.info("Storing the result locally...")
        test_acc_2_vote = torch.tensor(np.concatenate(test_acc_2_vote))
        test_acc_3_vote = torch.tensor(np.concatenate(test_acc_3_vote))
        test_acc_5_vote = torch.tensor(np.concatenate(test_acc_5_vote))
        self.validation_results = {
            "acc_2": acc_2,
            "acc_3": acc_3,
            "acc_5": acc_5,
            "Acc@2": test_acc_2_vote,
            "Acc@3": test_acc_3_vote,
            "Acc@5": test_acc_5_vote
        }

    def give_votes(self, logger, request_id, metric_to_observe):
        """
        Submit a vote for the given request.
        
        Parameters:
            logger: The logger object.
            request_id: The ID of the request.
            metric_to_observe: The metric to observe.
        """
        metrics = [metric_to_observe]
        for metric in metrics:
            vote_seq = self.validation_results[metric]
            self.vote(bool(vote_seq[request_id]))
    
    def aggr_decisions(self, logger):
        """
        Aggregate the decisions from all validators using the coordinator.
        
        Returns:
            bool: The aggregated decision.
        """
        if self.coordinator:
            val_result = self.coordinator.finalize()
            self.coordinator.reset_votes()
            return val_result
        return False
