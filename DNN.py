import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as Seq

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout_rate
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_batchnorm = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_batchnorms = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_batchnorms.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        # Input layer
        x = F.relu(self.input_batchnorm(self.input_layer(x)))
        print(x.shape)
        x = F.dropout(x, self.dropout, self.training)
        
        # Hidden layers
        for layer, batchnorm in zip(self.hidden_layers, self.hidden_batchnorms):
            x = F.relu(batchnorm(layer(x)))
            x = F.dropout(x, self.dropout, self.training)
        
        # Output layer
        x = self.output_layer(x)
        return x
