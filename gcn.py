import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def graph_convolution_fn(inputs, adj, weight, bias=None):
    batch_size = inputs.shape[0]
    support = torch.bmm(inputs, weight.expand(batch_size, -1, -1))
    output = torch.bmm(adj.expand(batch_size, -1, -1), support)
    if bias is not None:
        output = output + bias
    return output

def graph_diffusion_convolution_fn(inputs, adj, weight, bias=None, k=2):
    batch_size = inputs.shape[0]
    support = torch.bmm(inputs, weight.expand(batch_size, -1, -1))
    output = support.clone()
    adj_ = adj
    for i in range(k):
        output += torch.bmm(adj_.expand(batch_size, -1, -1), support)
        adj_ = torch.spmm(adj_, adj)
    if bias is not None:
        output = output + bias
    return output

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features*1)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, inputs, adj):
        return graph_convolution_fn(inputs, adj, self.weight, self.bias)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphDiffusionConvolution(Module):
    def __init__(self, in_features, out_features, k=2, bias=True):
        super(GraphDiffusionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.k = k
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features*self.k)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, inputs, DW):
        return graph_convolution_fn(inputs, DW, self.weight, self.bias, self.k)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


GraphConvolution = GraphDiffusionConvolution


class GCGRUCell(Module):
    def __init__(self, in_features, hidden_features, bias=True, conv_fn=graph_diffusion_convolution_fn):
        super(GCGRUCell, self).__init__()
        self.conv_fn = conv_fn
        self.in_features = in_features
        self.hidden_features = hidden_features
        
        self.weight_i = Parameter(torch.FloatTensor(self.in_features, 3*self.hidden_features))
        self.weight_h = Parameter(torch.FloatTensor(self.hidden_features, 3*self.hidden_features))
        self.bias = bias
        if bias:
            self.bias_i = Parameter(torch.FloatTensor(3*hidden_features))
            self.bias_h = Parameter(torch.FloatTensor(3*hidden_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv_i = 1. / math.sqrt(self.hidden_features*self.in_features)
        self.weight_i.data.uniform_(-stdv_i, stdv_i)
        stdv_h = 1. / math.sqrt(self.hidden_features*self.in_features)
        self.weight_h.data.uniform_(-stdv_h, stdv_h)
        if self.bias is not None:
            self.bias_i.data.uniform_(-stdv_i, stdv_i)
            self.bias_h.data.uniform_(-stdv_h, stdv_h)
            
    def check_forward_hidden(self, x, hx, hidden_label=''):
        if x.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    x.size(0), hidden_label, hx.size(0)))

        if hx.size(2) != self.hidden_features:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_features: got {}, expected {}".format(
                    hidden_label, hx.size(2), self.hidden_features))
            
    def forward(self, x, adj, hx=None):
        if hx is None:
            hx = x.new_zeros(x.size(0), x.size(1), self.hidden_features, requires_grad=False)
        self.check_forward_hidden(x, hx)
        
        gi = self.conv_fn(x, adj, self.weight_i, self.bias_i)
        gh = self.conv_fn(hx, adj, self.weight_h, self.bias_h)
        
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hx - newgate)
        return hy
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.hidden_features) + ')'
       
        
class GCGRU(Module):
    def __init__(self, in_features, hidden_features, bias=True, conv_fn=graph_diffusion_convolution_fn):
        super(GCGRU, self).__init__()
        self.gru_cell = GCGRUCell(in_features, hidden_features, bias, conv_fn)

    def forward(self, xs_seq, adj, all_out=True):
        r = []
        h = self.gru_cell(xs_seq[0], adj)
        r.append(h)
        for xs in xs_seq[1:]:
            r.append(self.gru_cell(xs_seq[0], adj, h))
        if all_out:
            return r
        else:
            return r[-1]
        

        
#import numpy as np
#x = torch.FloatTensor(10, 2, 24, 1)
#adj = torch.FloatTensor(np.eye(24))
#net = GCGRU(1, 4)
