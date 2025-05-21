import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='UCI',
                        choices=['UCI', 'Enron', 'GDELT', 'Memo-Tx'])
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--model_name', type=str, default='SGC', help='name of the model, note that EdgeBank is only applicable for evaluation',
                        choices=['GCN', 'GAT', 'SAGE', 'MLP', 'SGC', 'DeSocial'])
    parser.add_argument('--cuda', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=48, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=2, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'random_with_collision_check', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--use_feature', type=str, default='', help='whether to use text embeddings as feature') # or Bert
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--strategy', type=str, default='full', choices=['fine', 'full'], help='fine-tune or full-retrain')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--use_pretrained', action='store_true', default=False, help='whether to use pretrained model')
    parser.add_argument('--in_dim', type=int, default=384, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=128, help='output dimension')
    parser.add_argument('--start_period', type=int, default=0, help='start time')
    parser.add_argument('--end_period', type=int, default=38, help='end time')
    parser.add_argument('--name_tag', type=str, default='DeSocial-test', help='name tag (distinguish different experiments)')
    parser.add_argument('--test_interval_periods', type=int, default=10, help='test interval periods')

    ## Multi-Node Consensus Voting
    parser.add_argument('--experts', type=int, default=5, help='number of experts')

    ## Heuristics Backbone Selections
    parser.add_argument('--selection_mechanism', type=str, default='auto', choices=['random', 'rule', 'auto'], help='selection mechanism')
    parser.add_argument('--num_neighbor_samples', type=int, default=3000, help='number of neighbor samples')
    parser.add_argument('--alpha', type=float, default=0, help='alpha for time-based weighting decay')
    parser.add_argument('--f_pool', type=str, default="SGC+SAGE+MLP+GCN+GAT", help='backbone selection pool')

    ## Web3 Infrastructure
    parser.add_argument('--provider_url', type=str, default='http://127.0.0.1:7545', help='provider url')
    parser.add_argument('--contract_json_path', type=str, default='./contract/build/contracts/DeSocial.json', help='contract json')
    
    ## Evaluation Metrics
    ## We reported the run time based on observing one evaluation metric because the overload of voting and aggregation is high in serial, not parallel.
    parser.add_argument('--metric', type=str, default="Acc@2", help='Evaluation metric to observe' )

    args = parser.parse_args()
    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)
    
    return args

def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == "MLP":
        if args.dataset_name == "UCI":
            args.learning_rate = 0.005
        elif args.dataset_name == "Memo-Tx":
            args.learning_rate = 0.001
        elif args.dataset_name == "Enron":
            args.learning_rate = 0.0005
        elif args.dataset_name == "GDELT":
            args.learning_rate = 0.0001
        args.dropout = 0.7
    elif args.model_name == "SAGE":
        if args.dataset_name == "UCI":
            args.learning_rate = 0.001
        elif args.dataset_name == "Memo-Tx":
            args.learning_rate = 0.001
        elif args.dataset_name == "Enron":
            args.learning_rate = 0.0005
        elif args.dataset_name == "GDELT":
            args.learning_rate = 0.0005
        args.dropout = 0.7
    elif args.model_name == "GAT":
        if args.dataset_name == "UCI":
            args.learning_rate = 0.0001
        elif args.dataset_name == "Memo-Tx":
            args.learning_rate = 0.0001
        elif args.dataset_name == "Enron":
            args.learning_rate = 0.0001
        elif args.dataset_name == "GDELT":
            args.learning_rate = 0.0001
        args.dropout = 0.7
    elif args.model_name == "GCN":
        if args.dataset_name == "UCI":
            args.learning_rate = 0.0001
        elif args.dataset_name == "Memo-Tx":
            args.learning_rate = 0.0001
        elif args.dataset_name == "Enron":
            args.learning_rate = 0.0005
        elif args.dataset_name == "GDELT":
            args.learning_rate = 0.0005
        args.dropout = 0.7
    elif args.model_name == "SGC":
        if args.dataset_name == "UCI":
            args.learning_rate = 0.005
        elif args.dataset_name == "Memo-Tx":
            args.learning_rate = 0.0005
        elif args.dataset_name == "Enron":
            args.learning_rate = 0.005
        elif args.dataset_name == "GDELT":
            args.learning_rate = 0.001
        args.dropout = 0.7
    else:
        if args.model_name == "DeSocial":
            pass
        else:
            raise ValueError(f"Wrong model name {args.model_name}!")

def load_lr_given_models(model_name: str, dataset_name: str):

    if model_name == "MLP":
        if dataset_name == "UCI":
            return 0.005
        elif dataset_name == "Memo-Tx":
            return 0.001
        elif dataset_name == "Enron":
            return 0.0005
        elif dataset_name == "GDELT":
            return 0.0001
    elif model_name == "GCN":
        if dataset_name == "UCI":
            return 0.0001
        elif dataset_name == "Memo-Tx":
            return 0.0001
        elif dataset_name == "Enron":
            return 0.0005
        elif dataset_name == "GDELT":
            return 0.0005
    elif model_name == "GAT":
        if dataset_name == "UCI":
            return 0.0001
        elif dataset_name == "Memo-Tx":
            return 0.0001
        elif dataset_name == "Enron":
            return 0.0001
        elif dataset_name == "GDELT":
            return 0.0001
    elif model_name == "SAGE":
        if dataset_name == "UCI":
            return 0.001
        elif dataset_name == "Memo-Tx":
            return 0.001
        elif dataset_name == "Enron":
            return 0.0005
        elif dataset_name == "GDELT":
            return 0.0005
    elif model_name == "SGC":
        if dataset_name == "UCI":
            return 0.005
        elif dataset_name == "Memo-Tx":
            return 0.0005
        elif dataset_name == "Enron":
            return 0.005
        elif dataset_name == "GDELT":
            return 0.001

def get_num_users(dataset_name: str):
    if dataset_name == "UCI":
        return 1899
    elif dataset_name == "Memo-Tx":
        return 10907
    elif dataset_name == "Enron":
        return 42711
    elif dataset_name == "GDELT":
        return 6786
    else:
        raise ValueError(f"Wrong dataset name {dataset_name}!")