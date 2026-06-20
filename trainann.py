import torch
import torch.nn as nn

class ConcreteANN(nn.Module):
    def __init__(self, input_dim=8, hidden_dims=[128, 64, 32], dropout=0.2):
        super(ConcreteANN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
