import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from GSR_Net.initializations import *
from GSR_Net.preprocessing import normalize_adj_torch



class GSRLayer(nn.Module):
  
  def __init__(self,hr_dim):
    super(GSRLayer, self).__init__()
    self.device = torch.device('cuda')
    #self.weights = weight_variable_glorot(hr_dim)
    #self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True)
    initial_weights = weight_variable_glorot(hr_dim)
    self.weights = torch.nn.Parameter(data=torch.tensor(initial_weights, device=self.device, dtype=torch.float), requires_grad=True)
    
  def forward(self,A,X):
    lr = A
    lr_dim = lr.shape[0]
    f = X
    #eig_val_lr, U_lr = torch.symeig(lr, eigenvectors=True,upper=True)
    eig_val_lr, U_lr = torch.linalg.eigh(A, UPLO='U')
    U_lr = U_lr.to(X.device)
    # U_lr = torch.abs(U_lr)
    #eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
    eye_mat = torch.eye(lr_dim, device=self.device, dtype=torch.float)
    
    #zero_matrix = torch.zeros((160, 108)).to(X.device)

    s_d = torch.cat((eye_mat,eye_mat),0)  # it was eye_mat,eye_mat
    
    a = torch.matmul(self.weights,s_d )
    b = torch.matmul(a ,torch.t(U_lr))
    f_d = torch.matmul(b ,f)
    f_d = torch.abs(f_d)
    self.f_d = f_d.fill_diagonal_(1)
    adj = normalize_adj_torch(self.f_d)
    X = torch.mm(adj, adj.t())
    X = (X + X.t())/2
    idx = torch.eye(320, dtype=bool)  
    X[idx]=1
    return adj, torch.abs(X)
    


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    #160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.device = torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        #self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # input = F.dropout(input, self.dropout, self.training)
        input = input.to(self.weight.device)
        adj = adj.to(self.weight.device)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        # output = self.act(output)
        return output