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

from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import NegativeEdgeSampler
from eval import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics, get_vote_avg_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args, load_link_prediction_best_configs, load_lr_given_models, get_num_users
from model.models import GCN, GAT, SAGE, MLP, SGC

def assign_valid_groups(x):
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
    args.save_model_name = f'{args.model_name}_{args.name_tag}_{args.use_feature}-TEST'
    model_name_with_params = f"{args.save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-{args.strategy}"

    save_model_name = f'{args.model_name}_{args.name_tag}_{args.use_feature}'
    save_model_name_with_params = f"{save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-{args.strategy}"

    log_dir = f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{model_name_with_params}/"
    log_dir1 = f"./logs/{args.model_name}/{args.dataset_name}/{save_model_name}/{save_model_name_with_params}/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"{log_dir}/{model_name_with_params}0504.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    K_s = [3, 5, 10]

    vote_acc_mean = np.zeros((40,args.experts))
    avg_acc_mean = np.zeros((40,args.experts))
    vote_acc_std = np.zeros((40,args.experts))
    avg_acc_std = np.zeros((40,args.experts))

    vote_hit2_mean = np.zeros((40,args.experts,len(K_s)))
    avg_hit2_mean = np.zeros((40,args.experts,len(K_s)))
    vote_hit2_std = np.zeros((40,args.experts,len(K_s)))
    avg_hit2_std = np.zeros((40,args.experts,len(K_s)))

    vote_hit3_mean = np.zeros((40,args.experts,len(K_s)))
    avg_hit3_mean = np.zeros((40,args.experts,len(K_s)))
    vote_hit3_std = np.zeros((40,args.experts,len(K_s)))
    avg_hit3_std = np.zeros((40,args.experts,len(K_s)))
    
    models = []
    optimizers = []
    num_users = get_num_users(args.dataset_name)
    num_nodes = num_users
    model_type, model_nodes = assign_valid_groups(num_nodes)

    logger.info("Preparing models...")
    for i in tqdm(range(num_nodes)):
        node_model_type = model_type[i]
        node_model_type = args.model_name
        if node_model_type == "GCN":
            model = GCN(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "GAT":
            model = GAT(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "SAGE":
            model = SAGE(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "MLP":
            model = MLP(in_channels=args.in_dim, hidden_channels=args.hidden_dim, out_channels=args.out_dim)
        elif node_model_type == "SGC":
            model = SGC(in_channels=args.in_dim, out_channels=args.out_dim, K=2)
        
        if args.load_best_configs:
            node_lr = load_lr_given_models(model_name=node_model_type, 
                                       dataset_name=args.dataset_name)
        else:
            node_lr = args.learning_rate
        
        optimizer = create_optimizer(model=model, 
                                     optimizer_name=args.optimizer, 
                                     learning_rate=node_lr, 
                                     weight_decay=args.weight_decay)
        models.append(model)
        optimizers.append(optimizer)
    
    loss_func = nn.BCELoss()

    args.start_period = 28
    args.end_period = 38

    # Loading results...
    test_votes_all_time = torch.load(log_dir1+"test_votes_all_time.pt")
    test_preds_all_time = torch.load(log_dir1+"test_preds_all_time.pt")
    #test_preds_all_time = []
    #test_votes_all_time = []
    test_ranks2_all_time = []
    test_ranks3_all_time = []

    # TODO: Specify subgraph structure of each predicted link src node.
    # TODO: Compare direct testing result, whether identical to the trained result.
    selected_backbones_all = []
    for t in range(args.start_period, args.end_period):
        test_votes_all_time.append({})
        test_preds_all_time.append({})
        test_ranks2_all_time.append({})
        test_ranks3_all_time.append({})

        logger.info(f"Time Period: {t + 1}")
        logger.info(f"Testing Period: {t + 2}")
        
        if args.strategy == 'fine':
            node_raw_features, full_data, train_data, val_data, test_data = \
                get_link_prediction_data(dataset_name=args.dataset_name, args = args, train_time = t, val_time= (t + 1), test_time= (t + 2))
        else:
            node_raw_features, full_data, train_data, val_data, test_data = \
                get_link_prediction_data(dataset_name=args.dataset_name, args = args, train_time = 0, val_time= (t + 1), test_time= (t + 2))

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
        
        #logger.info("Users are selecting their models...")
        #backbones = backbone_selection(train_data, val_data, test_data)
        #selected_backbones_all.append(backbones)
        #print(backbones)

        test_votes_all = []
        test_ranks2_all = []
        test_ranks3_all = []
        test_acc = []
        test_ranks2_single = []
        test_ranks3_single = []

        # Getting validators...
        #validators = [100 * i for i in range(args.experts)]
        validators = [test_votes_all_time[t - args.start_period][i][0] for i in range(args.experts)]

        for i in range(args.experts):
            validator = int(validators[i])
            logger.info(f"Expert {i}: {validator}")
            print(f"Expert {i}: {validator}")

            if args.use_feature == "Text":
                x = torch.tensor(node_raw_features, dtype=torch.float)
            else:
                x = torch.tensor(node_raw_features, dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index).to(device=args.device)

            load_model_name_with_params = f"{save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-order={i}-t={t+1}-{args.strategy}"
            load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{save_model_name}/{load_model_name_with_params}/"
            pt_name = load_model_name_with_params = f"{save_model_name}-{args.dataset_name}-lr={args.learning_rate}-experts={args.experts}-{args.strategy}.pt"
            load_model_path = os.path.join(load_model_folder, pt_name)

            model = models[validator].to(args.device)
            model.load_state_dict(torch.load(load_model_path))
            model.eval()
            optimizer = optimizers[validator]
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

            test_losses, test_metrics, test_preds, test_votes, test_ranks2, test_ranks3 = evaluate_model_link_prediction(model_name=args.model_name, model=model, neighbor_sampler=None,
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
            models[validator] = model
            optimizers[validator] = optimizer
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

        for exp in [5]:
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

    logger.info("Training finished.")
    logger.info("Saving results...")
    #np.savez(f"{log_dir}/vote_acc_all.npz", mean=vote_acc_mean, std=vote_acc_std)
    #np.savez(f"{log_dir}/avg_acc_all.npz", mean=avg_acc_mean, std=avg_acc_std)
    #np.savez(f"{log_dir}/vote_hit2_all.npz", mean=vote_hit2_mean, std=vote_hit2_std)
    #np.savez(f"{log_dir}/avg_hit2_all.npz", mean=avg_hit2_mean, std=avg_hit2_std)
    #np.savez(f"{log_dir}/vote_hit3_all.npz", mean=vote_hit3_mean, std=vote_hit3_std)
    #np.savez(f"{log_dir}/avg_hit3_all.npz", mean=avg_hit3_mean, std=avg_hit3_std)
    np.savez(f"{log_dir}/overall.npz", vote_hit1_mean=vote_acc_mean, vote_hit1_std=vote_acc_std, 
             vote_hit2_mean=vote_hit2_mean, vote_hit2_std=vote_hit2_std, 
             vote_hit3_mean=vote_hit3_mean, vote_hit3_std=vote_hit3_std,
             avg_hit1_mean=avg_acc_mean, avg_hit1_std=avg_acc_std, 
             avg_hit2_mean=avg_hit2_mean, avg_hit2_std=avg_hit2_std,
             avg_hit3_mean=avg_hit3_mean, avg_hit3_std=avg_hit3_std)
    torch.save(test_votes_all_time, f"{log_dir}/test_votes_all_time1.pt")
    torch.save(test_preds_all_time, f"{log_dir}/test_preds_all_time1.pt")
    torch.save(test_ranks2_all_time, f"{log_dir}/test_ranks2_all_time.pt")
    torch.save(test_ranks3_all_time, f"{log_dir}/test_ranks3_all_time.pt")
    
    logger.info("Testing finished.")
    logger.info("Printing results...")
    for exp in [5]:
        logger.info(f"[Experts={exp}] Test avg acc: {np.round(np.mean(avg_acc_mean[29:39, exp]),4)} +- {np.round(np.std(avg_acc_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test vote acc: {np.round(np.mean(vote_acc_mean[29:39, exp]),4)} +- {np.round(np.std(vote_acc_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test acc gain: {np.round((np.mean(vote_acc_mean[29:39, exp]) - np.mean(avg_acc_mean[29:39, exp])) / np.mean(avg_acc_mean[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test avg H2: {np.round(np.mean(avg_hit2_mean[29:39, exp]),4)} +- {np.round(np.std(avg_hit2_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test vote H2 acc: {np.round(np.mean(vote_hit2_mean[29:39, exp]),4)} +- {np.round(np.std(vote_hit2_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test H2 gain: {np.round((np.mean(vote_hit2_mean[29:39, exp]) - np.mean(avg_hit2_mean[29:39, exp])) / np.mean(avg_hit2_mean[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test avg H3: {np.round(np.mean(avg_hit3_mean[29:39, exp]),4)} +- {np.round(np.std(avg_hit3_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test vote H3 acc: {np.round(np.mean(vote_hit3_mean[29:39, exp]),4)} +- {np.round(np.std(vote_hit3_std[29:39, exp]),4)}")
        logger.info(f"[Experts={exp}] Test H3 gain: {np.round((np.mean(vote_hit3_mean[29:39, exp]) - np.mean(avg_hit3_mean[29:39, exp])) / np.mean(avg_hit2_mean[29:39, exp]),4)}")
        
    sys.exit()
