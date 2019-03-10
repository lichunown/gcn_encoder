from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from gcn import GraphConvolution, GCGRUCell, GCGRU


class Encoder(nn.Module):
    def __init__(self, in_features=1, in_dim=24, out_features=8, out_dim=5, dropout=0.3):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(in_features, (out_features - in_features + 1)//2)
        self.gc2 = GraphConvolution((out_features - in_features + 1)//2, out_features)
        
        self.dence1 = nn.Linear(in_dim*out_features, 32)
        self.dence2 = nn.Linear(32, 16)
        self.dence3 = nn.Linear(16, out_dim)
        
        self.out_features = out_features
        self.in_dim = in_dim
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        x = x.view(-1, self.out_features*self.in_dim)
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        out = torch.sigmoid(self.dence3(x))
        return out

    
class Decoder(nn.Module):
    def __init__(self, in_dim=5, out_dim=24, out_features=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(in_dim, 8)
        self.dence2 = nn.Linear(8, 16)
        self.dence3 = nn.Linear(16, out_dim*16)
        self.out_dim = out_dim
        self.gc1 = GraphConvolution(16, 8)
        self.gc2 = GraphConvolution(8, out_features)
        
    def forward(self, x, adj):
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = F.relu(self.dence3(x))
        x = x.view(-1, self.out_dim, 16)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out = torch.sigmoid(self.gc2(x, adj))
        return out


class Encoder2(nn.Module):
    def __init__(self, in_features=1, in_dim=24, out_features=8, out_dim=5, dropout=0.3):
        super(Encoder2, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(in_features*in_dim, 110)
        self.dence2 = nn.Linear(110, 32)
        self.dence3 = nn.Linear(32, 16)
        self.dence4 = nn.Linear(16, out_dim)
        
    def forward(self, x, adj):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = F.relu(self.dence3(x))
        out = torch.sigmoid(self.dence4(x))
        return out

    
class Decoder2(nn.Module):
    def __init__(self, in_features=1, in_dim=24, out_features=8, out_dim=5, dropout=0.3):
        super(Decoder2, self).__init__()
        self.dropout = dropout
        self.dence1 = nn.Linear(out_dim, in_dim*in_features*4)
        self.dence2 = nn.Linear(in_dim*in_features*4, in_dim*in_features*2)
        self.dence3 = nn.Linear(in_dim*in_features*2, in_dim*in_features)
        self.dence4 = nn.Linear(in_dim*in_features, in_dim*in_features)
        self.in_dim = in_dim
        self.in_features = in_features
    def forward(self, x, adj):
        x = F.relu(self.dence1(x))
        x = F.relu(self.dence2(x))
        x = torch.sigmoid(self.dence3(x))
        x = torch.sigmoid(self.dence4(x))
        x = x.view(-1, self.in_dim, self.in_features)
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
    def __init__(self, in_features=1, in_dim=24, out_features=8, out_dim=5, dropout=0.3):
        super(Trainer2, self).__init__()
        self.encoder = Encoder2(in_features=in_features, in_dim=in_dim, out_features=out_features, out_dim=out_dim, dropout=dropout)
        self.decoder = Decoder2(in_features=in_features, in_dim=in_dim, out_features=out_features, out_dim=out_dim, dropout=dropout)
        
    def forward(self, x, adj, invadj):
        encode = self.encoder(x, adj)
        out = self.decoder(encode, invadj)
        return out

class Trainer3(nn.Module):
    def __init__(self, in_features=1, in_dim=24, out_features=8, out_dim=5, dropout=0.3):
        super(Trainer3, self).__init__()
        self.encoder = Encoder(in_features=in_features, in_dim=in_dim, out_features=out_features, out_dim=out_dim, dropout=dropout)
        self.decoder = Decoder2(in_features=in_features, in_dim=in_dim, out_features=out_features, out_dim=out_dim, dropout=dropout)
        
    def forward(self, x, adj, invadj):
        encode = self.encoder(x, adj)
        out = self.decoder(encode, invadj)
        return out
    

class Encoder_RNN(nn.Module):
    def __init__(self, channel_num_list=[1, 4, 8], dense_num_list=[(4+8)*24, 5], nodes_nums=24, dropout=0.):
        super(Encoder_RNN, self).__init__()
        if sum(channel_num_list[1:])*nodes_nums != dense_num_list[0]:
            raise ValueError("sum(channel_num_list[1:])*nodes_nums != dense_num_list[0]")
        self.gr_list = []
        for i in range(len(channel_num_list) - 1):
            self.gr_list.append(GCGRU(channel_num_list[i], channel_num_list[i + 1]))
        self.channel_num_list = channel_num_list
        self.dense_num_list = dense_num_list
        self.dense_list = []
        for i in range(len(dense_num_list) - 1):
            self.dense_list.append(nn.Linear(dense_num_list[i], dense_num_list[i + 1]))
            
        for i, layer in enumerate(self.gr_list):
            self.add_module('gr_list_'+str(i), layer)
        for i, layer in enumerate(self.dense_list):
            self.add_module('dense_list_'+str(i), layer)
            
    def forward(self, xs_seq, adj):
        # input shape (seqs, batch, nodes_features, channels)
        input_list = xs_seq
        gr_result = []
        for gr in self.gr_list:
            input_list = gr(input_list, adj)
            gr_result.append(input_list[-1])
        xs = torch.cat(gr_result, 2)
        
        batch_size = xs.size(0)
        xs = xs.view(batch_size, -1)
        for dense in self.dense_list[:-1]:
            xs = F.relu(dense(xs))
        xs = torch.sigmoid(self.dense_list[-1](xs))
        return xs
    
        
        
class Decoder_RNN(nn.Module):
    def __init__(self, channel_num_list=[8, 4, 1], dense_num_list=[5, (8+4)*24], seq_len = 10, nodes_nums=24, dropout=0.):
        super(Decoder_RNN, self).__init__()
        if sum(channel_num_list[:-1])*nodes_nums != dense_num_list[-1]:
            raise ValueError("sum(channel_num_list[:-1])*nodes_nums != dense_num_list[-1]")
        self.gr_list = []
        self.gr_list.append(GCGRU(channel_num_list[0], channel_num_list[0]))
        for i in range(len(channel_num_list) - 1):
            self.gr_list.append(GCGRU(channel_num_list[i], channel_num_list[i + 1]))
        self.channel_num_list = channel_num_list
        self.dense_num_list = dense_num_list
        self.dense_list = []
        for i in range(len(dense_num_list) - 1):
            self.dense_list.append(nn.Linear(dense_num_list[i], dense_num_list[i + 1]))
        
        self.seq_len = seq_len
        self.eos = Parameter(torch.FloatTensor(nodes_nums, channel_num_list[0]))
        self.reset_parameters()
        
        self.tm = self.channel_num_list[:-1]
        self.tm.reverse()
        
        self.change_outc2inc = GraphConvolution(channel_num_list[-1], channel_num_list[0])
        
        for i, layer in enumerate(self.gr_list):
            self.add_module('gr_list_'+str(i), layer)
        for i, layer in enumerate(self.dense_list):
            self.add_module('dense_list_'+str(i), layer)
            
    def reset_parameters(self):
        stdeos = 1
        self.eos.data.uniform_(-stdeos, stdeos)
        
    def forward(self, xs, adj):
        batch_size = xs.size(0)
        for dense in self.dense_list[:-1]:
            xs = F.relu(dense(xs))
        xs = torch.sigmoid(self.dense_list[-1](xs))
        xs = xs.view(batch_size, -1, sum(self.channel_num_list[:-1]))
        hs = list(xs.split(self.tm, 2))
        hs.reverse()
        hs.append(None)
        
        result = []
        for _ in range(self.seq_len):
            inputs = self.eos.expand(batch_size, -1, -1)
            next_hs = []
            for i, gr in enumerate(self.gr_list):
                gr = gr.gru_cell
                h = gr(inputs, adj, hs[i])
                next_hs.append(h)
                inputs = h
            result.append(inputs)
        
        return torch.stack(result)


class Trainer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, origin, adj):
        encode = self.encoder(origin, adj)
        if adj is not None:
            adj = adj.transpose(1, 0)
        decode = self.decoder(encode, adj)
        return decode
        
        
        
#import numpy as np
#x = torch.FloatTensor(np.random.random([100, 2, 24, 1]))
#adj = torch.FloatTensor(np.eye(24))
#net = Encoder_RNN()
#encode = net(x, adj)
#print(np.any(np.isnan(encode.data.numpy())))
#net2 = Decoder_RNN()
#y = net2(encode, adj.transpose(1, 0))
#y = y.data.numpy()
#print(np.any(np.isnan(y)))
