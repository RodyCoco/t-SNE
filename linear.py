from torch import nn
import torch

'''
input_dim: (features, actions)
output_dim: Tuple(timestamp, features)
'''
class Network(nn.Module):
    
    def __init__(self,  N, low_dim, high_dim, hidden_dim=[10, 8, 6]):
        super(Network, self).__init__()
        self.N = N
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.sigma = nn.Parameter(torch.ones(N*low_dim))
        self.sigma.requires_grad = False
        
        self.low_dim_linear = nn.Sequential(
            nn.Linear(N*low_dim, hidden_dim[0]),
            nn.PReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.PReLU(),
        )
        self.high_dim_linear = nn.Sequential(
            nn.Linear(N*high_dim, hidden_dim[0]),
            nn.PReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.PReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(2*hidden_dim[1], hidden_dim[2]),
            nn.PReLU(),
            nn.Linear(hidden_dim[2], N*low_dim),
            nn.Sigmoid(),
        )
        
        
    def forward(self, data):
        
        if len(data.shape) ==2:
            low_dim_data = data[:,:self.N*self.low_dim]
            high_dim_data = data[:,self.N*self.low_dim:]
        elif len(data.shape) == 3:
            low_dim_data = data[:, :,:self.N*self.low_dim]
            high_dim_data = data[:, :,self.N*self.low_dim:]
        low_hidden = self.low_dim_linear(low_dim_data)
        high_hidden = self.high_dim_linear(high_dim_data)
        hidden = torch.cat((low_hidden, high_hidden), dim=-1)
        out = self.linear(hidden)
        out = (out-0.5)*2 # range:0~1 -> -1~1
        return out