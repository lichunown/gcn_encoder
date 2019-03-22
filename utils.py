import numpy as np
import json
import networkx as nx
import pandas as pd
import torch
from functools import reduce
from torch.nn import functional as F

mul_all = lambda lists: reduce(lambda x,y:x*y, lists)

class VAELoss():
    def __call__(self, x, y, mu, logvar, dim=[0]):
        # print(x - y)
        # BCE = torch.sqrt(torch.mean((x-y)**2))
        BCE = F.binary_cross_entropy(y, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        print('RMSE loss: ', torch.sqrt(torch.mean((x-y)**2)), 'KLD loss', KLD)
        return (BCE + 1e-10 * KLD)


class RMSE():
    def __call__(self, x, y, dim=[0]):
        div_nums = mul_all([x.shape[i] for i in dim])
        return torch.sqrt(torch.sum((x-y)**2)) / div_nums


class MSE():
    def __call__(self, x, y, dim=[0]):
        div_nums = mul_all([x.shape[i] for i in dim])
        return torch.sum((x-y)**2) / div_nums

class MAPE():
    def __call__(self, x, y):
        return torch.mean(torch.sum(torch.sum((torch.log(x.add(1)) - torch.log(y.add(1)))**2, -1), -1))

def normalize(data):
    min_ = np.min(data, 0)
    max_ = np.max(data, 0)
    data = (data - min_)/(max_ - min_)
    return data  
