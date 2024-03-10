import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, Upsample
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import numpy as np
from torch_geometric.data import Data
from torch.autograd import Variable

N_SOURCE_NODES = 160
N_TARGET_NODES = 268

class Aligner(torch.nn.Module):
    def __init__(self):
        
        super(Aligner, self).__init__()

        nn = Sequential(Linear(1, N_SOURCE_NODES*N_SOURCE_NODES), ReLU())
        self.conv1 = NNConv(N_SOURCE_NODES, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv2 = NNConv(N_SOURCE_NODES, 1, nn, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        nn = Sequential(Linear(1, N_SOURCE_NODES), ReLU())
        self.conv3 = NNConv(1, N_SOURCE_NODES, nn, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(N_SOURCE_NODES, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        x1 = F.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)

        x2 = F.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([F.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:N_SOURCE_NODES]
        x5 = x3[:, N_SOURCE_NODES:2*N_SOURCE_NODES]

        x6 = (x4 + x5) / 2
        return x6

def create_batch(X, A):
    data_list = []
    for x, adj in zip(X, A):
        edge_index = adj.nonzero().t()
        edge_weights = adj[edge_index[0], edge_index[1]]
        edge_index, edge_weights = torch_geometric.utils.add_self_loops(edge_index, edge_weights) # add self connections
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weights.view(-1, 1))
        data_list.append(data)
    return torch_geometric.data.Batch().from_data_list(data_list)

def convert_generated_to_graph(data):
    """
        convert generated output from G to a graph
    """

    dataset = []

    for data in data1:
        counter = 0
        N_ROI = N_TARGET_NODES
        pos_edge_index = torch.zeros(2, N_ROI * N_ROI, dtype=torch.long)
        for i in range(N_ROI):
            for j in range(N_ROI):
                pos_edge_index[:, counter] = torch.tensor([i, j])
                counter += 1

        x = data
        pos_edge_index = torch.tensor(pos_edge_index, dtype=torch.long)
        data = Data(x=x, pos_edge_index= pos_edge_index, edge_attr=data.view(N_TARGET_NODES**2, 1))
        dataset.append(data)

    return dataset

