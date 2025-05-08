from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import torch

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)

def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False,
                             num_workers=2)
    return data_loader

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = np.unique(np.hstack((src_node_ids, dst_node_ids)))
        self.num_unique_nodes = len(self.unique_node_ids)

def get_link_prediction_data(dataset_name: str, args = None, train_time = 0, val_time = 1, test_time = 2):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_time: float, validation data time
    :param test_time: float, test data time
    :return: node_raw_features, full_data, train_data, val_data, test_data
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./data/{}/edge_list_new.csv'.format(dataset_name))
    
    # random node features
    node_raw_features = np.load(f'./data/{dataset_name}/node_feat.npy')

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times)

    # the setting of seed follows previous works
    random.seed(2020)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times < val_time, node_interact_times >= train_time)
    val_mask = np.logical_and(node_interact_times < test_time, node_interact_times >= val_time)
    test_mask = node_interact_times == test_time

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask])
    
    return node_raw_features, full_data, train_data, val_data, test_data
