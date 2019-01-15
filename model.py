import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn import GraphConvolution


class Encoder(nn.Module):
    def __init__(self, dropout=0.3):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(1, 4)
        self.gc2 = GraphConvolution(4, 8)
        
        self.dence1 = nn.Linear(24*8, 32)
        self.dence2 = nn.Linear(32, 16)
        self.dence3 = nn.Linear(16, 5)
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        b, nodes, channel = x.shape
        x = x.view(-1, nodes*channel)
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        out = F.sigmoid(self.dence3(x))
        return out

    
class Decoder(nn.Module):
    def __init__(self, dropout=0.3):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(5, 8)
        self.dence2 = nn.Linear(8, 16)
        self.dence3 = nn.Linear(16, 24*16)
        
        self.gc1 = GraphConvolution(16, 8)
        self.gc2 = GraphConvolution(8, 1)
        
    def forward(self, x, adj):
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = F.relu(self.dence3(x))
        x = x.view(-1, 24, 16)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out = F.sigmoid(self.gc2(x, adj))
        return out


class Encoder2(nn.Module):
    def __init__(self, dropout=0.3):
        super(Encoder2, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(24, 110)
        self.dence2 = nn.Linear(110, 32)
        self.dence3 = nn.Linear(32, 16)
        self.dence4 = nn.Linear(16, 5)
        
    def forward(self, x, adj):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = F.relu(self.dence3(x))
        out = F.sigmoid(self.dence4(x))
        return out

    
class Decoder2(nn.Module):
    def __init__(self, dropout=0.3):
        super(Decoder2, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(5, 16)
        self.dence2 = nn.Linear(16, 32)
        self.dence3 = nn.Linear(32, 24)
        self.dence4 = nn.Linear(24, 24)
        
    def forward(self, x, adj):
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = F.sigmoid(self.dence3(x))
        x = F.sigmoid(self.dence4(x))
        x = x.view(-1, 24, 1)
        return x
    
class Trainer1(nn.Module):
    def __init__(self, dropout=0.3):
        super(Trainer1, self).__init__()
        self.encoder = Encoder(dropout)
        self.decoder = Decoder(dropout)
        
    def forward(self, x, adj, invadj):
        encode = self.encoder(x, adj)
        out = self.decoder(encode, invadj)
        return out
    
class Trainer2(nn.Module):
    def __init__(self, dropout=0.3):
        super(Trainer2, self).__init__()
        self.encoder = Encoder2(dropout)
        self.decoder = Decoder2(dropout)
        
    def forward(self, x, adj, invadj):
        encode = self.encoder(x, adj)
        out = self.decoder(encode, invadj)
        return out

class Trainer3(nn.Module):
    def __init__(self, dropout=0.3):
        super(Trainer3, self).__init__()
        self.encoder = Encoder(dropout)
        self.decoder = Decoder2(dropout)
        
    def forward(self, x, adj, invadj):
        encode = self.encoder(x, adj)
        out = self.decoder(encode, invadj)
        return out