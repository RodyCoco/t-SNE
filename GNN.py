import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class NN_MessagePassingLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim, aggr='mean'):
        super(NN_MessagePassingLayer, self).__init__()
        self.aggr = aggr

        self.messageNN = nn.Linear(input_dim * 2, hidden_dim)
        self.updateNN = nn.Linear(input_dim + hidden_dim, output_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x, messageNN=self.messageNN, updateNN=self.updateNN)

    def message(self, x_i, x_j, messageNN):
        return messageNN(torch.cat((x_i, x_j), dim=-1))

    def update(self, aggr_out, x, updateNN):
        return updateNN(torch.cat((x, aggr_out), dim=-1))
    
class myGNN(torch.nn.Module):
    def __init__(self, var_dim, layer_num, input_dim, hidden_dim, output_dim, aggr='mean', var=-4.6, **kwargs):
        super(myGNN, self).__init__()
        self.var = nn.Parameter(torch.ones(var_dim)*var)
        self.var.requires_grad = True
        self.layer_num = layer_num
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # you can use the message passing layer you like, such as GCN, GAT, ...... 
        self.mp_layer = NN_MessagePassingLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                               output_dim=hidden_dim, aggr=aggr)

        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.PReLU1 = nn.PReLU()
        self.PReLU_list = [nn.PReLU(),nn.PReLU(),nn.PReLU()]
        
    def forward(self, x, edge_index):
        x = self.PReLU1(self.encoder(x))
        for i in range(self.layer_num):
            x = self.PReLU1(self.mp_layer(x, edge_index))
        node_out = (self.decoder(x).sigmoid()-0.5)*2
        return node_out