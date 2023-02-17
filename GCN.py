import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, var_dim, input_dim, hidden_dim, output_dim, var=-4.6,):
        super().__init__()
        self.var = nn.Parameter(torch.ones(var_dim)*var)
        self.var.requires_grad = True
        self.conv1 = torch_geometric.nn.GCNConv(input_dim, hidden_dim[0], aggr='mean')
        self.conv2 = torch_geometric.nn.GCNConv(hidden_dim[0], hidden_dim[1], aggr='mean')
        self.conv3 = torch_geometric.nn.GCNConv(hidden_dim[1], output_dim, aggr='mean')
        self.Sigmoid = nn.Sigmoid()
        self.PReLU1 = nn.PReLU()
        self.PReLU2 = nn.PReLU()
        
    def forward(self, x, edge_index, edge_attr):

        x = self.conv1(x, edge_index, edge_attr)
        x = self.PReLU1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.PReLU2(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = (self.Sigmoid(x)-0.5)*2
        return x