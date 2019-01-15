import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

#    def forward(self, inputs, adj):
#        r = []
#        for x in inputs:
#            support = torch.mm(x, self.weight)
#            output = torch.spmm(adj, support)
#            if self.bias is not None:
#                output = output + self.bias
#            output = output.reshape(1, *output.shape)
#            r.append(output)
#        return torch.cat(r)
    
    def forward(self, inputs, adj):
        batch_size = inputs.shape[0]
        support = torch.bmm(inputs, self.weight.expand(batch_size, -1, -1))
        output = torch.bmm(adj.expand(batch_size, -1, -1), support)
        if self.bias is not None:
            output = output + self.bias
        return output
    
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
        self.reset_parameters()
        self.k = k
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

#    def forward(self, inputs, adj):
#        r = []
#        for x in inputs:
#            support = torch.mm(x, self.weight)
#            output = torch.spmm(adj, support)
#            if self.bias is not None:
#                output = output + self.bias
#            output = output.reshape(1, *output.shape)
#            r.append(output)
#        return torch.cat(r)
    
    def forward(self, inputs, DW):
        batch_size = inputs.shape[0]
        support = torch.bmm(inputs, self.weight.expand(batch_size, -1, -1))
        output = support.clone()
        DW_ = DW
        for i in range(self.k):
            output += torch.bmm(DW_.expand(batch_size, -1, -1), support)
            DW_ = torch.spmm(DW_, DW)
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

GraphConvolution = GraphDiffusionConvolution
#class GCN(nn.Module):
#    def __init__(self, nfeat, nhid, nclass, dropout):
#        super(GCN, self).__init__()
#
#        self.gc1 = GraphConvolution(nfeat, nhid)
#        self.gc2 = GraphConvolution(nhid, nclass)
#        self.dropout = dropout
#
#    def forward(self, x, adj):
#        x = F.relu(self.gc1(x, adj))
#        x = F.dropout(x, self.dropout, training=self.training)
#        x = self.gc2(x, adj)
#        return F.log_softmax(x, dim=1)