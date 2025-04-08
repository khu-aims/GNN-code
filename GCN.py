import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn 
from ENCODER import ENCODER

class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, hgc, encoder, ehidden, conv_type, K, e_dropout_rate):
        super(GCN, self).__init__()
        hidden = hgc
        self.dropout = dropout
        bias = True 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.lg = len(hgc) 
        self.gconv = nn.ModuleList()
        self.e_dropout_rate = e_dropout_rate

        for i in range(len(hgc)):
            in_channels = input_dim if i==0  else hidden[i-1]
            if conv_type == "ChebConv":
                self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias)) 
            elif conv_type == "GCNConv":
                self.gconv.append(tg.nn.GCNConv(in_channels, hidden[i], bias=bias)) 
        
        self.cls = nn.Sequential(
                torch.nn.Linear(hidden[-1], 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.edge_net = ENCODER(input_dim=edgenet_input_dim//2, dropout=dropout, encoder = encoder, hidden = ehidden)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x, edge_index, edgenet_input): 
        edge_weight = self.edge_net(edgenet_input)

        if self.e_dropout_rate > 0 :
            edge_index, edge_mask = tg.utils.dropout_edge(edge_index, p=self.e_dropout_rate, training= self.training)
            edge_weight = edge_weight[edge_mask]

        h = x
        for i in range(0, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight)) 

        logit = self.cls(h) 

        return logit
