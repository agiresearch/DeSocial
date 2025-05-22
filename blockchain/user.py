from web3 import Web3
import numpy as np
from utils.DataLoader import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.utils import set_random_seed, create_optimizer
from utils.utils import NegativeEdgeSampler
from utils.DataLoader import Data as MyData
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, load_link_prediction_best_configs, load_lr_given_models, get_num_users
from model.dispatcher import Dispatcher
from eval import evaluate
import os
from tqdm import tqdm
import time

# global lists, storing all the role objects.
user_storage = []
address_id_map = {}

class BC_User:

    def __init__(self, user_id, bc_address, num_of_val, url, web3, num_nodes):
        """
            Here gives a brief description of the arguments.
            Web3 Properties:
                bc_address: the address of the user on the blockchain.
                num_of_val: the number of validators in the system.
                url: the url of the blockchain.
                web3: the web3 object.
                contract: the smart contract object.
            
            Local Data Storage:
                user_id: the id of the user.
                model: the model object.
                train_data: the training data.
                val_data: the validation data.
                test_data: the test data.
                requests_collected: the requests collected from the source nodes.
        """

        ## Web3 Properties
        self.bc_address = bc_address
        self.url = url
        self.web3 = web3
        self.contract = None # will initialize in run.py
        self.num_of_val = num_of_val
        
        ## Local Data Storage
        self.user_id = user_id
        self.model = None # will initialize in run.py
        self.optimizer = None # will initialize in run.py
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.requests_collected = []
        self.pa_requesters = []
        self.pos_neighbor = []
        self.neg_neighbor = []
        self.neighbor_weight = []
        self.task_result = []
        self.task_results_to_comp = []
        self.validation_results = {}
        self.num_nodes = num_nodes
        
        # User Backbone Selection
        self.neighbor_sample_list = []
    
    def send_a_request(self, target, inter_terminal, timestamp):
        """
            The initiator sends a request to the blokchain that it wants to add an edge to the target node.
            Parameters:
                target: the target node.
                inter_terminal: the intermediate terminal node.
                timestamp: the timestamp of the request.
        """
        # the user acts as an initiator, sending a request to solidity by `request(target, inter_terminal)`
        # in request function, the smart contract collects the request.
        while True:
            tx = self.contract.functions.request(user_storage[target].bc_address, int(inter_terminal)).transact({
                "from": self.bc_address,
                "gas": 300000,
                "gasPrice": self.web3.to_wei("20", "gwei"), # to make the transaction more efficient, set the gas price to 20 gwei
            })
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
            # the smart contract recognized this request will be stored in the intermediate terminal node.
            # the intermediate terminal node will store the request in its local data storage.
            if tx_receipt["status"] == 1:
                break

    def retrieve_test_data(self, inter_terminal):
        """
            Signed by the validator.
            The validator retrieves the test data from the intermediate terminal node.
            Parameters:
                inter_terminal: the intermediate terminal node.
        """
        self.contract.functions.retrieve(int(inter_terminal)).call()
        self.test_data = user_storage[inter_terminal].test_data

    def broadcast(self, inter_terminal):
        """
            Signed by the receiver.
            After validating all the tasks in predicting the next preiod, all the ground truth edges will be broadcast to everyone, preparing for the next period's experiment.
            Step:
                1. the receiver signs the retrieval.
                2. the receiver copies the ground truth edges to its local data storage.
        """
        while True:
            tx = self.contract.functions.broadcast(inter_terminal).transact({
                "from": self.bc_address,
                "gas": 300000,
                "gasPrice": self.web3.to_wei("20", "gwei")
            })
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
            if tx_receipt["status"] == 1:
                # user_storage[target].test_data
                self.update_social_network()
                # if the transaction is successful, the ground truth edges are broadcasted to everyone.
                self.test_data = user_storage[inter_terminal].test_data
                break

    def select_validators(self, val_tot, val_num):
        """
            the intermediate terminal node request the blockchain to choose validator randomly.
            Parameters:
                val_tot: the total number of validators in a specified backbone community.
                val_num: the number of validators to be chosen.
            Return:
                validators: the list of validator indices.
        """
        while True:
            tx = self.contract.functions.select_validators(val_tot, val_num).transact({
                "from": self.bc_address,
                "gas": 300000,
                "gasPrice": self.web3.to_wei("20", "gwei")
            })
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
            if tx_receipt["status"] == 1:
                validators = self.contract.functions.get_val_list().call()
                break
        return validators

    def vote(self, agree):
        """
            Each validator give their votes based on their backbone.
        """
        tx = self.contract.functions.vote(self.bc_address, agree).transact({
            "from": self.bc_address,
            "gas": 300000,
            "gasPrice": self.web3.to_wei("20", "gwei")
        })
        tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)

    def update_social_network(self):
        """
            Two steps:
                1. merge the old train data and the old val data to new train data.
                2. the new val data is the test data.
        """
        src_node_ids = np.concatenate((self.train_data.src_node_ids, self.val_data.src_node_ids), axis=0)
        dst_node_ids = np.concatenate((self.train_data.dst_node_ids, self.val_data.dst_node_ids), axis=0)
        node_interact_times = np.concatenate((self.train_data.node_interact_times, self.val_data.node_interact_times), axis=0)
        self.train_data = MyData(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)
        self.val_data = self.test_data
    
    # Here is the machine learning part.
    # The user will train the model and update their own model.
    def give_prediction(self, logger, args, save_model_folder, model_name_with_params):
        """
            The validator will give the prediction of the next period.
            The validator will train the model and update their own model.
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

        ## Here stores the voting results to blockchain.
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
            For every requests, the validator will give their votes, by calling the smart contract.
        """
        # Here the validator will give their votes.
        metrics = [metric_to_observe]
        for metric in metrics:
            vote_seq = self.validation_results[metric]
            self.vote(bool(vote_seq[request_id]))
    
    def aggr_decisions(self, logger):
        """
            The blockchain will aggregate the decisions of the validators.
            The blockchain will call the smart contract to aggregate the decisions.
        """
        while True:
            tx = self.contract.functions.finalize().transact({
                "from": self.bc_address,
                "gas": 300000,
                "gasPrice": self.web3.to_wei("20", "gwei")
            })
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx)
            if tx_receipt["status"] == 1:
                val_result = self.contract.functions.get_FINAL_RESULT.call()
                return val_result