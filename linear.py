from torch import nn
import torch

'''
input_dim: (features, actions)
output_dim: Tuple(timestamp, features)
'''
class Network(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim = [100, 60, 40]):
        super(Network, self).__init__()
        self.output_dim = output_dim
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.PReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.PReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.PReLU(),
        )
        self.linear4 = nn.Sequential(
            nn.Linear(hidden_dim[2], self.output_dim),
            nn.PReLU(),
        )
        self.residual_linear1 = nn.Linear(input_dim, hidden_dim[1])
        self.residual_linear2 = nn.Linear(input_dim, hidden_dim[2])
        
    def forward(self, in_data):
        hidden = self.linear1(in_data)
        hidden = self.linear2(hidden)
        hidden = self.linear3(hidden)
        out = self.linear4(hidden)

        return out