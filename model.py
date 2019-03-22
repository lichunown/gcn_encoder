from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from gcn import GraphConvolution, GCGRUCell, GCGRU

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
        self.dense_list1 = []
        self.dense_list2 = []
        for i in range(len(dense_num_list) - 1):
            self.dense_list1.append(nn.Linear(dense_num_list[i], dense_num_list[i + 1]))
            self.dense_list2.append(nn.Linear(dense_num_list[i], dense_num_list[i + 1]))
            
        for i, layer in enumerate(self.gr_list):
            self.add_module('gr_list_'+str(i), layer)
        for i, layer in enumerate(self.dense_list1):
            self.add_module('dense_list1_'+str(i), layer)
        for i, layer in enumerate(self.dense_list2):
            self.add_module('dense_list2_'+str(i), layer)
        self.sampleNumber = 50
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = 0
        for i in range(self.sampleNumber):
            eps += torch.randn_like(std)
        eps = eps / self.sampleNumber
        return mu + eps * std
            
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

        for dense in self.dense_list1[:-1]:
            xs = F.relu(dense(xs))
        mu = torch.sigmoid(self.dense_list1[-1](xs))

        for dense in self.dense_list2[:-1]:
            xs = F.relu(dense(xs))
        logvar = torch.sigmoid(self.dense_list2[-1](xs))

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar
    
        
        
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
                h = F.relu(gr(inputs, adj, hs[i]))
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
        encode, mu, logvar = self.encoder(origin, adj)
        if adj is not None:
            adj = adj.transpose(1, 0)
        decode = self.decoder(encode, adj)
        return decode, mu, logvar
        
        

