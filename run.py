import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.utils import set_random_seed, create_optimizer
from utils.utils import NegativeEdgeSampler
from utils.metrics import get_link_prediction_metrics_train, get_vote_avg_metrics
from utils.DataLoader import Data
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, load_link_prediction_best_configs, load_lr_given_models, get_num_users
from model.dispatcher import Dispatcher
from eval import evaluate

from blockchain.user import BC_User
from blockchain.blockchain import Blockchain
from blockchain.user import user_storage, address_id_map

def retreive_test_data(full_data, test_time):
    """
        Retreive the test data for time `test_time`.
    """
    src_node_ids = full_data.src_node_ids
    dst_node_ids = full_data.dst_node_ids
    node_interact_times = full_data.node_interact_times
    node_weights = full_data.node_weights
    test_mask = node_interact_times == test_time
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], node_weights=node_weights[test_mask])
    return test_data

def assign_valid_groups(x):
    """
        Assign each node to a valid group.
    """
    model_name = ['MLP', 'GCN', 'GAT', 'SAGE', 'SGC']
    num_of_model_types = len(model_name)
    base = x // num_of_model_types
    extra = x % num_of_model_types
    result = []
    model_nodes = {}
    for part in range(num_of_model_types):
        count = base + (1 if part < extra else 0)
        model_nodes[model_name[part]] = list(range(len(result), len(result) + count))
        result.extend([model_name[part]] * count)
    return result, model_nodes

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    args = get_link_prediction_args(is_evaluation=False)
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    if args.load_best_configs:
        print("Loading best configurations...")
        load_link_prediction_best_configs(args=args)
    
    prog_start_time = time.time()
    args.save_model_name = f'{args.model_name}_{args.name_tag}_{args.use_feature}'
    model_name_with_params = f"{args.save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-{args.strategy}"
    log_dir = f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{model_name_with_params}/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"{log_dir}/{model_name_with_params}-{args.start_period}-{args.end_period}.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    vote_acc_mean = np.zeros((40,args.experts))
    avg_acc_mean = np.zeros((40,args.experts))
    vote_acc_std = np.zeros((40,args.experts))
    avg_acc_std = np.zeros((40,args.experts))
    vote_hit2_mean = np.zeros((40,args.experts))
    avg_hit2_mean = np.zeros((40,args.experts))
    vote_hit2_std = np.zeros((40,args.experts))
    avg_hit2_std = np.zeros((40,args.experts))
    vote_hit3_mean = np.zeros((40,args.experts))
    avg_hit3_mean = np.zeros((40,args.experts))
    vote_hit3_std = np.zeros((40,args.experts))
    avg_hit3_std = np.zeros((40,args.experts))
    test_preds_all_time = []
    test_votes_all_time = []
    test_ranks2_all_time = []
    test_ranks3_all_time = []
    
    # Initialize the blockchain.
    num_users = get_num_users(args.dataset_name)
    num_nodes = num_users
    model_type, model_nodes = assign_valid_groups(num_nodes)
    blockchain = Blockchain(provider_url=args.provider_url, contract_json_path=args.contract_json_path, num_node=num_nodes)
    accounts = blockchain.web3.eth.accounts

    # Allocate memory for each user.
    logger.info("Initializing users in blockchain...")
    for i in tqdm(range(num_nodes)):
        node_model_type = model_type[i]
        model = Dispatcher(node_model_type, args=args)
        if args.load_best_configs:
            node_lr = load_lr_given_models(model_name=node_model_type, dataset_name=args.dataset_name)
        else:
            node_lr = args.learning_rate
        optimizer = create_optimizer(model=model, 
                                     optimizer_name=args.optimizer, 
                                     learning_rate=node_lr, 
                                     weight_decay=args.weight_decay)
        user = BC_User(user_id=i, 
                        bc_address=accounts[i], 
                        num_node=num_nodes,
                        url=args.provider_url,
                        web3=blockchain.web3)
        user.model = model
        user.optimizer = optimizer
        user_storage.append(user)
    
    loss_func = nn.BCELoss()

    # extract backbone models from f_pool
    backbone_models = args.f_pool.split('+')
    # Here construct each model list, stored locally.
    # When the smart contract selects validators, it will just return the index list, and we can directly use the list.
    validator_communities = []
    for i in range(len(backbone_models)):
        validator_communities.append(model_nodes[backbone_models[i]])

    # Initialize the graph data for first several periods.
    # Then add the edges one by one in the future periods.
    logger.info("Loading data for each user...")
    node_raw_features, full_data, train_data, val_data, test_data = \
        get_link_prediction_data(dataset_name = args.dataset_name, 
                                 args = args, 
                                 train_time = 0, 
                                 val_time = (args.start_period + 1), 
                                 test_time = (args.start_period + 2))
    for i in tqdm(range(num_nodes)):
        user_storage[i].node_raw_features = node_raw_features
        user_storage[i].train_data = train_data
        user_storage[i].val_data = val_data

    logger.info("Start training period by period...")
    for t in range(args.start_period, args.end_period):
        logger.info(f"Current Period: {t + 1}")
        logger.info(f"Test Period: {t + 2}")

        test_data = retreive_test_data(full_data, t + 2)
        test_data_requests = test_data.src_node_ids.shape[0]

        test_preds_all_time.append({})
        test_votes_all_time.append({})
        test_ranks2_all_time.append({})
        test_ranks3_all_time.append({})
        test_votes_all = []
        test_ranks2_all = []
        test_ranks3_all = []
        test_acc = []
        test_ranks2_single = []
        test_ranks3_single = []

        # deal with the future requests one by one on chain.
        # the requests will finally stored in the local location of the first requester. (to run the blockchain faster)
        inter_terminal = test_data.src_node_ids[0]
        #set_inter_terminal(blockchain, inter_terminal)
        request_time = []
        for i in range(test_data_requests):
            requester = int(test_data.src_node_ids[i])
            target = int(test_data.dst_node_ids[i])
            # the requester sends a request to the blockchain.
            request_start_time = time.time()
            user_storage[requester].send_a_request(target, inter_terminal, t+2)
            request_end_time = time.time()
            request_time.append(request_end_time - request_start_time)
        logger.info(f"Average request frequency: {np.round(1 / np.mean(request_time),2)} requests per seconds.")

        logger.info("The intermidiate terminal is packing all the requests...")
        # the intermidiate terminal first sort out all the requests, and then send to all the validators.
        # user_storage[inter_terminal].request_collected -> test_data
        requests = np.array(user_storage[inter_terminal].request_collected)
        test_src_id = requests[:, 0]
        test_dst_id = requests[:, 1]
        test_node_interact_times = requests[:, 2]
        test_data = Data(src_node_ids=test_src_id, dst_node_ids=test_dst_id, node_interact_times=test_node_interact_times)
        user_storage[inter_terminal].test_data = test_data

        # then the validators retrieve the test data from the intermidiate terminal by smart contract.
        # the validator will sign the receipt saying they have received the test data.
        for i in range(len(validator_communities)):
            for j in range(args.experts):
                validator = validator_selected[i][j]
                user_storage[validator].retrieve_test_data(validator, inter_terminal)

        logger.info("Selecting validators from each backbone model group...")
        validator_selected = []
        for i in range(len(validator_communities)):
            # in this stage, the intermidiate terminal send a request
            # to let the blockchain select validators for each backbone model, by returning random indices.
            val_indices = user_storage[inter_terminal].select_validators(len(validator_communities[i]), args.experts)
            validator_selected.append(validator_communities[i][val_indices])

        # Then the validators start training.
        logger.info("Each validator is training...")
        for i in range(len(validator_communities)):
            for j in range(args.experts):
                validator = validator_selected[i][j]
                logger.info(f"Expert {i}: {validator}")
                logger.info(f'{validator} is requesting test task package at time {t+2} from {inter_terminal}...')
                train_data = user_storage[validator].train_data
                val_data = user_storage[validator].val_data
                test_data = user_storage[validator].test_data

                edge_index = torch.tensor([train_data.src_node_ids, train_data.dst_node_ids], dtype=torch.long)
                edge_index_dir = edge_index
                edge_index_inv = torch.tensor([train_data.dst_node_ids, train_data.src_node_ids], dtype=torch.long)
                edge_index = torch.cat([edge_index, edge_index_inv], dim=1)

                train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), 
                                                            batch_size=args.batch_size, 
                                                            shuffle=True)
                val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), 
                                                        batch_size=args.batch_size, 
                                                        shuffle=False)
                test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), 
                                                        batch_size=args.batch_size, 
                                                        shuffle=False)

                x = torch.tensor(node_raw_features, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index).to(device=args.device)

                model = user_storage[validator].model.to(args.device)
                optimizer = user_storage[validator].optimizer
                set_random_seed(seed=validator)

                train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, 
                                                            dst_node_ids=train_data.dst_node_ids, 
                                                            interact_times=train_data.node_interact_times, 
                                                            negative_sample_strategy=args.negative_sample_strategy,
                                                            last_observed_time=t,
                                                            seed=validator)
                val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,   
                                                        dst_node_ids=full_data.dst_node_ids, 
                                                        interact_times=full_data.node_interact_times, 
                                                        negative_sample_strategy=args.negative_sample_strategy,
                                                        last_observed_time=t + 1,
                                                        seed=validator)
                test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,   
                                                            dst_node_ids=full_data.dst_node_ids, 
                                                            interact_times=full_data.node_interact_times, 
                                                            negative_sample_strategy=args.negative_sample_strategy,
                                                            last_observed_time=t + 2,
                                                            seed=validator)

                save_model_name_with_params = f"{args.save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-order={i}-t={t+1}-{args.strategy}"
                save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{save_model_name_with_params}/"
                os.makedirs(save_model_folder, exist_ok=True)
                early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder, save_model_name=model_name_with_params, logger=logger, model_name=args.model_name)

                for epoch in range(args.num_epochs):
                    model.train()
                    train_losses, train_metrics = [], []
                    for batch_indices in tqdm(train_idx_data_loader, ncols=120):
                        batch_indices = batch_indices.numpy()
                        src = torch.tensor(train_data.src_node_ids[batch_indices], device=args.device)
                        dst = torch.tensor(train_data.dst_node_ids[batch_indices], device=args.device)
                        if args.negative_sample_strategy != 'random':
                            _, neg_dst = train_neg_edge_sampler.sample(len(src), 
                                                                    batch_src_node_ids=src, 
                                                                    batch_dst_node_ids=dst, 
                                                                    current_batch_start_time=train_data.node_interact_times[batch_indices][0], 
                                                                    current_batch_end_time=train_data.node_interact_times[batch_indices][-1])
                        else:
                            _, neg_dst = train_neg_edge_sampler.sample(len(src))
                        neg_dst = torch.tensor(neg_dst, device=args.device)

                        edge_label_index = torch.cat([torch.stack([src, dst], dim=0), torch.stack([src, neg_dst], dim=0)], dim=1)
                        edge_label = torch.cat([torch.ones(src.size(0)), torch.zeros(src.size(0))]).to(args.device)

                        pred = model(data.to(args.device), edge_label_index).squeeze(dim=-1).sigmoid()
                        loss = loss_func(pred, edge_label)

                        train_losses.append(loss.item())
                        train_metrics.append(get_link_prediction_metrics_train(predicts=pred.sigmoid(), labels=edge_label))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    logger.info(f"Epoch {epoch + 1}: Train loss {np.mean(train_losses):.4f}")

                    if (epoch + 1) % args.test_interval_epochs == 0:
                        val_losses, val_metrics, _, _, _, _ = evaluate(model_name=args.model_name, model=model, neighbor_sampler=None,
                            evaluate_idx_data_loader=val_idx_data_loader, evaluate_neg_edge_sampler=val_neg_edge_sampler,
                            evaluate_data=val_data, loss_func=loss_func, num_neighbors=None, time_gap=None, data=data, device=args.device)

                        #val_metric_indicator = [(k, np.mean([m[k] for m in val_metrics]), True) for k in val_metrics[0].keys()]
                        val_metric_indicator = [('val loss', np.mean(val_losses), False)]
                        logger.info(f"val loss: {np.mean(val_losses)}")
                        early_stop = early_stopping.step(val_metric_indicator, model)
                        if early_stop:
                            break

                early_stopping.load_checkpoint(model)
                logger.info("Training finished. Best model loaded.")

                test_losses, test_metrics, test_preds, test_votes, test_ranks2, test_ranks3 = evaluate(model_name=args.model_name, model=model, neighbor_sampler=None,
                    evaluate_idx_data_loader=test_idx_data_loader, evaluate_neg_edge_sampler=test_neg_edge_sampler,
                    evaluate_data=test_data, loss_func=loss_func, num_neighbors=None, time_gap=None, data=data, device=args.device, is_test = True)
                logger.info(f"Final test loss: {np.mean(test_losses):.4f}")
                weights = np.array([m["weight"] for m in test_metrics])
                for metric in test_metrics[0].keys():
                    values = np.array([m[metric] for m in test_metrics])
                    metric_value = np.sum(values * weights) / np.sum(weights)
                    logger.info(f"Test {metric}: {metric_value:.4f}")
                    if metric == "accuracy":
                        acc = metric_value
                    elif metric == "H2":
                        hits2 = metric_value
                    elif metric == "H3":
                        hits3 = metric_value
                
                model = model.cpu()
                user_storage[validator].model = model
                user_storage[validator].optimizer = optimizer
                test_votes = torch.tensor(np.concatenate(test_votes))
                test_votes_all.append(test_votes)
                test_ranks2 = torch.tensor(np.concatenate(test_ranks2))
                test_ranks2_all.append(test_ranks2)
                test_ranks3 = torch.tensor(np.concatenate(test_ranks3))
                test_ranks3_all.append(test_ranks3)
                test_acc.append(acc)
                test_ranks2_single.append(hits2)
                test_ranks3_single.append(hits3)

                test_votes_all_time[-1][i] = (validator, test_votes)
                test_preds_all_time[-1][i] = (validator, test_preds)
                test_ranks2_all_time[-1][i] = (validator, test_ranks2)
                test_ranks3_all_time[-1][i] = (validator, test_ranks3)

            # done by the first p, on-chain finalization.
            # Here should be modified, calculate each validation task one by one.
            exp = args.experts
            avg_acc, avg_acc_s, vote_acc, vote_acc_s, avg_rank2, avg_rank2_s, vote_rank2_acc, vote_rank2_acc_s, avg_rank3, avg_rank3_s, vote_rank3_acc, vote_rank3_acc_s = get_vote_avg_metrics(test_votes_all, test_ranks2_all, test_ranks3_all, test_acc, test_ranks2_single, test_ranks3_single, exp)
            logger.info(f"[Experts={exp}] Final avg acc: {avg_acc} +- {avg_acc_s}")
            logger.info(f"[Experts={exp}] Final vote acc: {vote_acc} +- {vote_acc_s}")
            logger.info(f"[Experts={exp}] Final acc gain: {np.round((vote_acc - avg_acc) / avg_acc,4)}")
            logger.info(f"[Experts={exp}] Final avg rank2: {avg_rank2} +- {avg_rank2_s}")
            logger.info(f"[Experts={exp}] Final vote rank2 acc: {vote_rank2_acc} +- {vote_rank2_acc_s}")
            logger.info(f"[Experts={exp}] Final rank2 gain: {np.round((vote_rank2_acc - avg_rank2) / avg_rank2,4)}")
            logger.info(f"[Experts={exp}] Final avg rank3: {avg_rank3} +- {avg_rank3_s}")
            logger.info(f"[Experts={exp}] Final vote rank3 acc: {vote_rank3_acc} +- {vote_rank3_acc_s}")
            logger.info(f"[Experts={exp}] Final rank3 gain: {np.round((vote_rank3_acc - avg_rank3) / avg_rank3,4)}")
            
            vote_acc_mean[t+1][exp] = vote_acc
            avg_acc_mean[t+1][exp] = avg_acc
            avg_acc_std[t+1][exp] = avg_acc_s
            vote_acc_std[t+1][exp] = vote_acc_s
            vote_hit2_mean[t+1][exp] = vote_rank2_acc
            avg_hit2_mean[t+1][exp] = avg_rank2
            vote_hit2_std[t+1][exp] = vote_rank2_acc_s
            avg_hit2_std[t+1][exp] = avg_rank2_s
            vote_hit3_mean[t+1][exp] = vote_rank3_acc
            avg_hit3_mean[t+1][exp] = avg_rank3
            vote_hit3_std[t+1][exp] = vote_rank3_acc_s
            avg_hit3_std[t+1][exp] = avg_rank3_s

        if t != args.end_time:
            # all the nodes merge the previous train_data and val_data to train_data
            # letting the new val_data to test_data.
            node_raw_features, full_data, train_data, val_data, test_data = \
                get_link_prediction_data(dataset_name=args.dataset_name, args = args, train_time = 0, val_time= (t + 1), test_time= (t + 2))
        
    logger.info("Training finished.")
    logger.info("Saving results...")
    np.savez(f"{log_dir}/overall.npz", vote_hit1_mean=vote_acc_mean, vote_hit1_std=vote_acc_std, 
             vote_hit2_mean=vote_hit2_mean, vote_hit2_std=vote_hit2_std, 
             vote_hit3_mean=vote_hit3_mean, vote_hit3_std=vote_hit3_std,
             avg_hit1_mean=avg_acc_mean, avg_hit1_std=avg_acc_std, 
             avg_hit2_mean=avg_hit2_mean, avg_hit2_std=avg_hit2_std,
             avg_hit3_mean=avg_hit3_mean, avg_hit3_std=avg_hit3_std)
    torch.save(test_votes_all_time, f"{log_dir}/test_votes_all_time.pt")
    torch.save(test_preds_all_time, f"{log_dir}/test_preds_all_time.pt")
    torch.save(test_ranks2_all_time, f"{log_dir}/test_ranks2_all_time.pt")
    torch.save(test_ranks3_all_time, f"{log_dir}/test_ranks3_all_time.pt")
    
    logger.info("Printing results...")
    exp = args.experts
    logger.info(f"[Experts={exp}] Test avg acc: {np.round(np.mean(avg_acc_mean[29:39, exp]),4)} +- {np.round(np.std(avg_acc_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test vote acc: {np.round(np.mean(vote_acc_mean[29:39, exp]),4)} +- {np.round(np.std(vote_acc_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test gain: {np.round((np.mean(vote_acc_mean[29:39, exp]) - np.mean(avg_acc_mean[29:39, exp])) / np.mean(avg_acc_mean[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test avg H2: {np.round(np.mean(avg_hit2_mean[29:39, exp]),4)} +- {np.round(np.std(avg_hit2_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test vote H2 acc: {np.round(np.mean(vote_hit2_mean[29:39, exp]),4)} +- {np.round(np.std(vote_hit2_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test H2 gain: {np.round((np.mean(vote_hit2_mean[29:39, exp]) - np.mean(avg_hit2_mean[29:39, exp])) / np.mean(avg_hit2_mean[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test avg H3: {np.round(np.mean(avg_hit3_mean[29:39, exp]),4)} +- {np.round(np.std(avg_hit3_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test vote H3 acc: {np.round(np.mean(vote_hit3_mean[29:39, exp]),4)} +- {np.round(np.std(vote_hit3_std[29:39, exp]),4)}")
    logger.info(f"[Experts={exp}] Test H3 gain: {np.round((np.mean(vote_hit3_mean[29:39, exp]) - np.mean(avg_hit3_mean[29:39, exp])) / np.mean(avg_hit2_mean[29:39, exp]),4)}")

    sys.exit()
