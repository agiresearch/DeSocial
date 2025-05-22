import argparse

def get_link_prediction_args():
    """
        Get the args for the link prediction task
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='UCI',
                        choices=['UCI', 'Enron', 'GDELT', 'Memo-Tx'])
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--model_name', type=str, default='SAGE', help='name of the model',
                        choices=['GCN', 'GAT', 'SAGE', 'MLP', 'SGC'])
    parser.add_argument('--cuda', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--test_interval_epochs', type=int, default=2, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'random_with_collision_check', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--use_feature', type=str, default='', help='whether to use text embeddings as feature')
    parser.add_argument('--strategy', type=str, default='full', choices=['fine', 'full'], help='fine-tune or full-retrain')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--in_dim', type=int, default=384, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=128, help='output dimension')
    parser.add_argument('--start_period', type=int, default=0, help='start time')
    parser.add_argument('--end_period', type=int, default=38, help='end time')
    parser.add_argument('--name_tag', type=str, default='DeSocial-Run', help='name tag (distinguish different experiments)')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    ## Multi-Node Decentralized Consensus Voting
    parser.add_argument('--experts', type=int, default=5, help='number of experts')

    ## Heuristics Backbone Selections
    parser.add_argument('--selection_mechanism', type=str, default='auto', choices=['random', 'rule', 'auto'], help='selection mechanism')
    parser.add_argument('--num_neighbor_samples', type=int, default=750, help='number of neighbor samples') # gamma
    parser.add_argument('--alpha', type=float, default=-0.1, help='alpha for time-based weighting decay')
    parser.add_argument('--f_pool', type=str, default="SGC+SAGE+MLP+GCN+GAT", help='backbone selection pool')

    ## Web3 Infrastructure
    parser.add_argument('--provider_url', type=str, default='http://127.0.0.1:7545', help='provider url')
    parser.add_argument('--contract_json_path', type=str, default='./contract/build/contracts/DeSocial.json', help='contract json')
    
    ## Evaluation Metrics
    ## We reported the run time based on observing one evaluation metric because the overload of voting and aggregation is high in serial, not parallel.
    parser.add_argument('--metric', type=str, default="Acc@2", help='Evaluation metric to observe' )

    args = parser.parse_args()
    if args.load_best_configs:
        print("Loading best configurations...")
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
    
    if args.f_pool == "PA":

        if args.dataset_name == "UCI":
            if args.metric == "Acc@2":
                args.f_pool = "SGC+SAGE+MLP+GCN"
                args.num_neighbor_samples = 750
                args.alpha = -0.1
            elif args.metric == "Acc@3":
                args.f_pool = "SGC+MLP"
                args.num_neighbor_samples = 1250
                args.alpha = -0.1
            elif args.metric == "Acc@5":
                args.f_pool = "SGC+MLP"
                args.num_neighbor_samples = 1250
                args.alpha = -0.1
                
        elif args.dataset_name == "Memo-Tx":
            if args.metric == "Acc@2":
                args.f_pool = "SGC+SAGE"
                args.num_neighbor_samples = 250
                args.alpha = 0
            elif args.metric == "Acc@3":
                args.f_pool = "SGC+SAGE"
                args.num_neighbor_samples = 1000
                args.alpha = 0
            elif args.metric == "Acc@5":
                args.f_pool = "SGC+GCN"
                args.num_neighbor_samples = 1250
                args.alpha = -0.01
        
        elif args.dataset_name == "Enron":
            if args.metric == "Acc@2":
                args.f_pool = "SAGE+GAT"
                args.num_neighbor_samples = 1250
                args.alpha = -0.1
            elif args.metric == "Acc@3":
                args.f_pool = "SAGE+GAT"
                args.num_neighbor_samples = 1250
                args.alpha = -0.1
            elif args.metric == "Acc@5":
                args.f_pool = "SGC+SAGE+GAT"
                args.num_neighbor_samples = 750
                args.alpha = -0.1
                
        elif args.dataset_name == "GDELT":
            if args.metric == "Acc@2":
                args.f_pool = "SGC+SAGE+GAT"
                args.num_neighbor_samples = 750
                args.alpha = 0
            elif args.metric == "Acc@3":
                args.f_pool = "SGC+SAGE"
                args.num_neighbor_samples = 750
                args.alpha = 0
            elif args.metric == "Acc@5":
                args.f_pool = "SGC+GCN"
                args.num_neighbor_samples = 1000
                args.alpha = 0

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