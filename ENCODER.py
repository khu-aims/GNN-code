import torch
from torch.nn import Linear as Lin
from torch import nn

class ENCODER(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2, encoder='PAE', hidden = 64):
        super(ENCODER, self).__init__()
        self.encoder = encoder

        self.parser = nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True)
                )
        
        self.fcl = nn.Sequential(
                nn.Linear(hidden * 2, 1, bias=True),
                nn.Sigmoid()
                )
         
        self.ea =nn.Sequential(
                nn.Linear(hidden, 1, bias=True),
                nn.Sigmoid()
                )
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.L2 = nn.PairwiseDistance( eps=1e-8)
        self.sigmoid = nn.Sigmoid()
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU()

    def forward(self, x):

        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]

        h1 = self.parser(x1) 
        h2 = self.parser(x2) 
        
        if self.encoder == 'PAE':
            p = (self.cos(h1,h2) + 1) * 0.5

        elif self.encoder == "EA":
            p = abs(h1 - h2)
            p = self.ea(p)

        elif self.encoder == "Tanh":
            p = self.cos(h1,h2)
            p = self.relu(p)
            p = self.tanh(p)

        elif self.encoder == "exp":
            p = self.cos(h1,h2)
            p = torch.exp(p)

        elif self.encoder == 'L2':
            p = self.L2(h1, h2)
            p = 1/p
            p = self.sigmoid(p)


        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True



