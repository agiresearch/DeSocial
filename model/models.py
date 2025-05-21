import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, SGConv
from torch_geometric.nn.models import LightGCN
from torch_geometric.nn import GATv2Conv, BatchNorm

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index).sigmoid()

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, concat=False, dropout=0.7):
        super(GAT, self).__init__()

        self.concat = concat
        self.heads = heads
        self.dropout = dropout

        self.hidden_dim = hidden_channels * heads if concat else hidden_channels
        self.out_dim = out_channels * heads if concat else out_channels

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=concat, dropout=dropout)
        self.norm1 = BatchNorm(self.hidden_dim)

        self.conv2 = GATv2Conv(self.hidden_dim, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.norm2 = BatchNorm(self.out_dim)

        self.res_proj1 = nn.Linear(in_channels, self.hidden_dim) if in_channels != self.hidden_dim else nn.Identity()
        self.res_proj2 = nn.Linear(self.hidden_dim, self.out_dim) if self.hidden_dim != self.out_dim else nn.Identity()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        x_res = self.res_proj1(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x + x_res)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_res = self.res_proj2(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_res

        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index).sigmoid()

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def encode(self, data):
        x = data.x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.fc2(x)

        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

class LGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LGCN, self).__init__()
        self.encoder = LightGCN(in_channels, out_channels, num_layers = 2)
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        edge_index = data.edge_index
        z = self.encoder.get_embedding(edge_index)
        return self.decode(z, edge_label_index)

class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=K, cached=True)
        self.mlp = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1)
        )

    def encode(self, data):
        x = self.conv(data.x, data.edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = self.mlp(src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)