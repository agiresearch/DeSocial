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
from utils.utils import set_random_seed, create_optimizer
from utils.utils import assign_valid_groups, retreive_test_data
from utils.dataloader import Data
from utils.dataloader import get_idx_data_loader, get_link_prediction_data
from utils.configs import get_link_prediction_args, load_link_prediction_best_configs, load_lr_given_models, get_num_users
from model.dispatcher import Dispatcher
from model.select import Selection

from blockchain.user import BC_User
from blockchain.blockchain import Blockchain
from blockchain.user import user_storage

if __name__ == "__main__":
    
    pa_enabled = True

    warnings.filterwarnings('ignore')
    args = get_link_prediction_args()
    if "+" not in args.f_pool:
        pa_enabled = False
        args.model_name = args.f_pool
    
    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    prog_start_time = time.time()
    args.save_model_name = f'{args.model_name}_{args.name_tag}_{args.use_feature}'
    model_name_with_params = f"{args.save_model_name}-{args.dataset_name}-experts={args.experts}-{args.strategy}"
    log_dir = f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{model_name_with_params}/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/{model_name_with_params}-{args.start_period}-{args.end_period}.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    results = []
    
    ##########################################
    # The blockchain and user storage setup. # 
    ##########################################
    # Initialize the blockchain.
    num_users = get_num_users(args.dataset_name)
    num_nodes = num_users
    model_type, model_nodes = assign_valid_groups(num_nodes)
    blockchain = Blockchain(provider_url=args.provider_url, contract_json_path=args.contract_json_path, num_node=num_nodes)
    accounts = blockchain.web3.eth.accounts
    dispatcher = Dispatcher()

    # Allocate memory for each user.
    logger.info("Initializing users in blockchain...")
    for i in tqdm(range(num_nodes)):
        node_model_type = model_type[i]
        model = dispatcher.dispatch(node_model_type, args=args)
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
                       num_of_val=args.experts,
                       url=args.provider_url,
                       web3=blockchain.web3,
                       num_nodes=num_nodes)
        user.contract = blockchain.contract
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

    #####################
    # DeSocial Pipeline #
    #####################
    logger.info("Start training and testing period by period...")
    for t in range(args.start_period, args.end_period):
        logger.info(f"Current Period: {t + 1}")
        logger.info(f"Test Period: {t + 2}")

        test_data = retreive_test_data(full_data, t + 2)
        num_requests = test_data.src_node_ids.shape[0]
        logger.info(f'Number of requests this period: {num_requests}')

        # deal with the future requests one by one on chain.
        # the requests will finally stored in the local location of the first requester. (to run the blockchain faster)
        inter_terminal = test_data.src_node_ids[0]
        sel_val_time = []
        unique_src = np.unique(test_data.src_node_ids)
        num_of_request_src = unique_src.shape[0]
        
        request_start_time = time.time()
        for i in tqdm(range(num_requests)):
            #####################################################################################
            # Step 1. Specify the requester p and target q for each link prediction task (p,q). # 
            #####################################################################################
            requester = int(test_data.src_node_ids[i])
            target = int(test_data.dst_node_ids[i])
            
            #################################################################################################################
            # Step 2(1). The requester p sends the request to the blockchain, and the intermediate node store the requests. # 
            #################################################################################################################
            user_storage[requester].send_a_request(target, inter_terminal, t+2)
            user_storage[inter_terminal].requests_collected.append([requester, target, t+2])
        
        request_end_time = time.time()
        request_time = request_end_time - request_start_time
        logger.info(f"Amortized Step 2 (Request) time: {np.round(request_time / num_of_request_src, 4)}.")
        
        ########################################################################
        # Step 2(2). The intermediate terminal is collecting all the requests. #
        ########################################################################
        logger.info("The intermediate terminal is collecting all the requests...")
        requests = np.array(user_storage[inter_terminal].requests_collected)
        test_src_id = requests[:, 0]
        test_dst_id = requests[:, 1]
        test_node_interact_times = requests[:, 2]
        test_data = Data(src_node_ids=test_src_id, dst_node_ids=test_dst_id, node_interact_times=test_node_interact_times)
        user_storage[inter_terminal].test_data = test_data
        
        #####################################################################################################
        # Step 2(3). The blockchain selects validators from each community (the group with same backbones). #
        #####################################################################################################
        logger.info("Selecting validators from each backbone model group...")
        validator_selected = []
        sel_val_start_time = time.time()
        for i in range(len(validator_communities)):
            logger.info(f"Selecting validators from nodes possess {backbone_models[i]}...")
            # in this stage, the intermidiate terminal send a request
            # to let the blockchain select validators for each backbone model, by returning random indices.
            val_indices = user_storage[inter_terminal].select_validators(len(validator_communities[i]), args.experts)
            validator_selected.append([validator_communities[i][v] for v in val_indices])
        
        sel_val_end_time = time.time()
        sel_val_time = sel_val_end_time - sel_val_start_time
        logger.info(f"Amortized Step 2 (Validator selection) time: {np.round(sel_val_time / (args.experts * len(validator_communities)), 4)}")
        print(validator_selected)
        
        # then the validators retrieve the test data from the intermidiate terminal by smart contract.
        # the validator will sign the receipt saying they have received the test data.
        retrieve_start_time = time.time()
        
        #################################################################################################################
        # Step 2(4). Each valdidator retrieving test requests from the intermediate node by calling the smart contract. #
        #################################################################################################################
        logger.info("Each validator is retrieving the test data...")
        for i in range(len(validator_communities)):
            for j in range(args.experts):
                validator = validator_selected[i][j]
                user_storage[validator].retrieve_test_data(inter_terminal)
                logger.info(f"Validator {validator} has received the test data from {inter_terminal}.")
        retrieve_end_time = time.time()
        retrieve_time = retrieve_end_time - retrieve_start_time
        logger.info(f"Amortized Step 2 (retrieving) time: {np.round(retrieve_time / (args.experts * len(validator_communities)), 4)}.")

        ##################################################
        # Step 3. Each validator trains their own model. #
        ##################################################
        logger.info("Each validator is training...")
        for i in range(len(validator_communities)):
            for j in range(args.experts):
                validator = validator_selected[i][j]
                logger.info(f"Expert {j} @ {backbone_models[i]}: {validator}")
                logger.info(f'{validator} is requesting test task package at time {t+2} from {inter_terminal}...')
                save_model_name_with_params = f"{args.save_model_name}-{backbone_models[i]}-{args.dataset_name}-experts={args.experts}-order={j}-t={t+1}-{args.strategy}"
                save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{save_model_name_with_params}/"
                os.makedirs(save_model_folder, exist_ok=True)
                user_storage[validator].give_prediction(logger, args, save_model_folder, model_name_with_params)
        
        if pa_enabled:

            #############################################################################
            # Step 4. Each requester p creates a historical neighborhood sampling task. #
            #############################################################################
            create_start_time = time.time()
            logger.info("Each requester is creating the historical neighborhood sampling task...")
            train_data_cur = user_storage[inter_terminal].train_data
            val_data_cur = user_storage[inter_terminal].val_data
            selection = Selection(train_data, 
                                val_data, 
                                t+2, 
                                num_nodes, 
                                num_of_request_src, 
                                node_raw_features, 
                                args.device, 
                                alpha=args.alpha, 
                                gamma=args.num_neighbor_samples)
            selection.neighbor_setup()
            
            for p in unique_src:
                pos_neighbor, neg_neighbor, neighbor_weight = selection.create_neighbor_samples(p)
                user_storage[p].pos_neighbor = pos_neighbor
                user_storage[p].neg_neighbor = neg_neighbor
                user_storage[p].neighbor_weight = neighbor_weight

            create_end_time = time.time()
            create_time = create_end_time - create_start_time
            logger.info(f"Amortized Step 4 (Create) time: {np.round(create_time / num_of_request_src, 4)}.")
            
            ###########################################################################################################
            # Step 5. Each requester p sends the request to evaluate the neighbor sampling task on various backbones. #
            ###########################################################################################################
            pos_neighbors = []
            neg_neighbors = []
            neighbor_weights = []
            pa5_start = time.time()
            for i in range(num_of_request_src):
                user_storage[unique_src[i]].send_a_request(0, inter_terminal, t+2)
                # the intermediate counts the requesters.
                user_storage[inter_terminal].pa_requesters.append(unique_src[i])
                pos_neighbors.extend(user_storage[unique_src[i]].pos_neighbor)
                neg_neighbors.extend(user_storage[unique_src[i]].neg_neighbor)
                neighbor_weights.extend(user_storage[unique_src[i]].neighbor_weight)

            pa5_end = time.time()
            pa5_time = pa5_end - pa5_start
            logger.info(f"Amortized Step 5 time: {np.round(pa5_time / num_of_request_src, 4)}.")

            ########################################################################
            # Step 6. One of the nodes in each validator community take the tasks. #
            ########################################################################
            for i in range(len(validator_communities)):
                validator = validator_selected[i][0]
                selection.backbones.append(validator)
            
            for backbone_id in range(len(selection.backbones)):
                validator = validator_selected[backbone_id][0]
                # each backbone model will take the tasks and return the results.
                user_storage[validator].task_result = selection.take_exam(pos_neighbors, neg_neighbors, neighbor_weights, validator)

            ######################################################################################
            # Step 7. The requesters are returned the test results from each candidate backbone. #
            ######################################################################################
            pa7_start = time.time()
            for i in range(num_of_request_src):
                user_storage[unique_src[i]].send_a_request(0, inter_terminal, t+2)
                for backbone_id in range(len(selection.backbones)):
                    validator = validator_selected[backbone_id][0]
                    user_storage[unique_src[i]].task_results_to_comp.append(user_storage[validator].task_result[i])
                    
            pa7_end = time.time()
            pa7_time = pa7_end - pa7_start
            logger.info(f"Amortized Step 7 time: {np.round(pa7_time / num_of_request_src, 4)}.")
            
            #############################################################
            # Step 8. The requester specify its personalized algorithm. #
            #############################################################
            logger.info("The requesters are selecting the best backbone model...")
            backbones = {}
            for i in range(num_of_request_src):
                # get the test results for each backbone model
                # and select the best one.
                task_result_array = np.array(user_storage[unique_src[i]].task_results_to_comp)
                # select the best backbone model
                best_backbone = np.argmax(task_result_array)
                backbones[unique_src[i]] = best_backbone

        ##########################################################################
        # Step 9. User give their votes and the blockchain aggregates the votes. #
        ##########################################################################
        logger.info("Final aggregation...")
        vote_start_time = time.time()
        decisions = []
        for i in tqdm(range(num_requests)):
            p = int(test_data.src_node_ids[i])
            q = int(test_data.dst_node_ids[i])
            # Setting the personalized algorithm for each request user p.
            # This is very important because the correct validator community need to give votes.
            if not pa_enabled:
                personalized_algorithm = 0
            else:
                personalized_algorithm = backbones[p]
            
            validator_p = validator_selected[personalized_algorithm]
            for j in range(args.experts):
                validator = validator_p[j]
                user_storage[validator].give_votes(logger, i, args.metric)
            # aggregate the votes
            decision = user_storage[inter_terminal].aggr_decisions(logger)
            decisions.append(decision)
            
        vote_end_time = time.time()
        vote_time = vote_end_time - vote_start_time
        logger.info(f"Step 9 (Vote and Aggregation) time: {np.round(vote_end_time - vote_start_time,4)}.")
        logger.info(f"Amortized Step 9 (Vote and Aggregation) time: {np.round(vote_time / num_of_request_src, 4)}.")
        
        exp = args.experts
        decisions_report = np.mean(decisions)
        logger.info(f"[Experts={exp}] Final vote {args.metric}: {np.round(decisions_report, 4)}")
        results.append(decisions_report)

        ###################################################################################
        # Step 10. All the nodes merge the previous train_data and val_data to train_data #
        ###################################################################################
        logger.info("Users are updating their social network data...")
        if t != args.end_period - 1:
            broadcast_start_time = time.time()
            user_storage[0].retrieve_test_data(inter_terminal)
            broadcast_end_time = time.time()
            broadcast_time = broadcast_end_time - broadcast_start_time
            logger.info(f"Amortized Step 10 - Call time: {np.round(broadcast_time / 1, 4)}.")
            broadcast_start_time = time.time()
            user_storage[0].update_social_network()
            broadcast_end_time = time.time()
            broadcast_time = broadcast_end_time - broadcast_start_time
            logger.info(f"Amortized Step 10 - Copy time: {np.round(broadcast_time / 1, 4)}.")
            
            for i in tqdm(range(1, num_nodes)):
                user_storage[i].train_data = user_storage[0].train_data
                user_storage[i].val_data = user_storage[0].val_data
        
        # Empty the storage of current period requests.
        user_storage[inter_terminal].requests_collected = []
        user_storage[inter_terminal].pa_requesters = []
        for p in unique_src:
            user_storage[p].pos_neighbor = []
            user_storage[p].neg_neighbor = []
            user_storage[p].neighbor_weight = []
            user_storage[p].task_results = []
            user_storage[p].task_results_to_comp = []
        
    logger.info("Testing finished.")
    logger.info("Printing results...")
    exp = args.experts
    results = np.array(results)
    logger.info(f"[Experts={exp}] Test {args.metric}: {np.round(np.mean(results), 4)}")

    sys.exit()
