import numpy as np
import json
import networkx as nx
import pandas as pd


def normalize(data):
    min_ = np.min(data, 0)
    max_ = np.max(data, 0)
    data = (data - min_)/(max_ - min_)
    return data
    
def load_G(path, n=6):
    with open(path, 'r') as f:
        weight = json.load(f)
    
    weight_values = np.array(list(weight.values())).reshape(-1)
    weight_values = (weight_values - np.min(weight_values))/(np.max(weight_values) - np.min(weight_values))
    for i, name in enumerate(weight):
        weight[name] = weight_values[i]
    
    edges = [(*e.split('-'), weight[e]) for e in weight]
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(n)])
    G.add_weighted_edges_from(edges)
    G.add_weighted_edges_from([(str(i), str(i), 1) for i in range(n)])
    
    A = nx.adjacency_matrix(G).todense()
    D = np.eye(n)
    D[list(range(n)),list(range(n))] = np.sum(A, 0)
    L = D - A
#    L = np.dot(np.dot(Ds, A), Ds)
    # N = D^{-1/2} L D^{-1/2}
    lplc = nx.normalized_laplacian_matrix(G).todense()
    return lplc, L, D, G


def read_data24():
    data_path = 'data/state.npy'
    weight_path = 'data/weight.json'
    
    n_start = 30
    n_end   = 53
    
    data = np.load(data_path)
    
    with open(weight_path, 'r') as f:
        weight = json.load(f)
    
    weight_values = np.array(list(weight.values()))
    weight_values = (weight_values - np.min(weight_values))/(np.max(weight_values) - np.min(weight_values))
    for i, name in enumerate(weight):
        weight[name] = weight_values[i]
    
    edges = [(*e.split('-'), weight[e]) for e in weight]
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(n_start, n_end+1)])
    G.add_weighted_edges_from(edges)
    G.add_weighted_edges_from([(str(i), str(i), 1) for i in range(n_start, n_end + 1)])
    
    A = nx.adjacency_matrix(G).todense()
    D = np.eye(n_end - n_start + 1)
    D[list(range(n_end - n_start + 1)),list(range(n_end - n_start + 1))] = np.sum(A, 0)
    L = D - A
#    L = np.dot(np.dot(Ds, A), Ds)
    # N = D^{-1/2} L D^{-1/2}
    lplc = nx.normalized_laplacian_matrix(G).todense()
    
    data = data[:, n_start:n_end + 1]
    std = np.std(data, 0)
    mean = np.mean(data, 0)
#    data = (data - mean) / (3*std)
    data = normalize(data)
    return G, D, L, lplc, data, std, mean


G24, D24, L24, lplc24, data24, std24, mean24 = read_data24()

data_artificial = np.load('data/artificialWithAnomaly.npy').T
data_artificial = normalize(data_artificial)
lplc_artificial, L_artificial, D_artificial, G_artificial = load_G('data/artificialWithAnomaly_correlation2.json')

def yield_data(data, batch_size, random=True):
    data = data.copy()
    max_len = len(data)
    if random:
        np.random.shuffle(data)
    i = 0
    while i + batch_size <= max_len:
        p = data[i: i + batch_size]
        yield p.reshape(*p.shape, 1)
        i += batch_size
    p = data[i:].reshape(*data.shape, 1)
    yield p.reshape(*p.shape, 1)
    
    
def yield_data_n(data, batch_size, n, random=True):
    data = data.copy()
    max_len = len(data)
    if random:
        np.random.shuffle(data)
        
    i = 0
    while i + batch_size*n <= max_len:
        t = data[i: i + batch_size*n].reshape(batch_size, n, -1)
        yield t.transpose(0, 2, 1)
        i += batch_size*n
    b = (max_len-i)//n
    t = data[i: i + b*n]
    if list(t):
        t = t.reshape(b, n, -1)
        yield t.transpose(0, 2, 1)
    
    

def yield_data_time(data, batch_size, n, random=True):
    data = data.copy()
    if random:
        np.random.shuffle(data)
    
    def yield_data(data, batch_size):
        for i in range(len(data) - batch_size):
            yield data[i: i + batch_size]
    
    r = []
    for p in yield_data(data, batch_size):
        r.append(p)
        if len(r)==n:
            yield np.array(r)
            r = []
        

import torch
from functools import reduce

mul_all = lambda lists: reduce(lambda x,y:x*y, lists)

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



