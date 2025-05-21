import random
import torch
import torch.nn as nn
import numpy as np

from utils.DataLoader import Data

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer

class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids = None, dst_node_ids = None, num_nodes: int = None,
                 negative_sample_strategy: str = 'random', seed: int = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param negative_sample_strategy: str, negative sampling strategy, we use "random" in our paper
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.num_nodes = num_nodes
        if type(src_node_ids) == np.ndarray:
            self.unique_src_node_ids = np.unique(src_node_ids)
            self.unique_dst_node_ids = np.unique(dst_node_ids)
        else:
            self.unique_src_node_ids = np.arange(self.num_nodes)
            self.unique_dst_node_ids = np.arange(self.num_nodes)
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size: int):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :return:
        """
        negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)

def retreive_test_data(full_data, test_time):
    """
        Retreive the test data for time `test_time`.
    """
    src_node_ids = full_data.src_node_ids
    dst_node_ids = full_data.dst_node_ids
    node_interact_times = full_data.node_interact_times
    test_mask = node_interact_times == test_time
    test_data = Data(src_node_ids=src_node_ids[test_mask], 
                     dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask])
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