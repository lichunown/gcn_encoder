import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

graph_data = np.load('data/new/result_fin.npy')
#graph_data = np.load('data/new/pearsonr.npy')
#graph_data = np.load('data/new/mutual.npy')
graph_data = graph_data + graph_data.T

graph_data[np.where(graph_data<0.1)] = 0

graph_data = 1 - graph_data
graph_data = graph_data - np.eye(len(graph_data))

graph_data = np.exp(graph_data*10)
#graph_data[np.where(graph_data>0.5)] = 1


info = np.load('data/new/info2.npy')


G = nx.Graph() 
G.add_nodes_from([str(i) for i in range(len(G))])
where = np.where(graph_data != 0)
G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], graph_data[where]))

#%%

nodes_list = np.arange(424)
dist_matrix = graph_data
k = 2

def kmeans(nodes_list, dist_matrix, k):
    select = random.sample(list(nodes_list), k)
    next_select_list = np.zeros(k, dtype='int')
    dist = np.zeros([len(nodes_list), k])
    
    while True:
        for i in range(k):
            dist[:, i] = dist_matrix[select[i], :]
        mindist = np.min(dist, 1)
        for i in range(k):
            ids = np.where(dist[:, i] == mindist)[0]
            next_select = []
            for id_ in ids:
                next_select.append(np.sum(dist_matrix[id_, ids]))
            next_select_id = np.argmin(next_select)
            next_select_list[i] = next_select_id
        
        if set(list(select))==set(list(next_select_list)):
            break
        else:
            select = next_select_list
            
    for i in range(k):
        dist[:, i] = dist_matrix[select[i], :]
    mindist = np.min(dist, 1)
    return np.argmax(dist == mindist.reshape(-1,1), 1), np.mean(dist,0)

kmeans(np.arange(len(graph_data)), graph_data, 3)

#%%
x = np.arange(2, 40)
mean_dist = []
for i in x:
    tmp = []
    print(i)
    for _ in range(1000):
        pick, dist = kmeans(np.arange(len(graph_data)), graph_data, i)
        tmp.append(np.mean(dist))
    mean_dist.append(np.mean(tmp))
    
plt.plot(x, mean_dist)

#%%
min_result = None
min_dist = np.inf
for _ in range(10000):
    pick, dist = kmeans(np.arange(len(graph_data)), graph_data, 10)
    if len(set(list(dist)))==10:
        if np.mean(dist) < min_dist:
            min_dist = np.mean(dist)
            min_result = pick

div = [[] for _ in range(10)]
for i, info_ in enumerate(info):
    div[min_result[i]].append(info_)
    
for i, div_ in enumerate(div):
    div[i] = np.array(div_)

for div_ in div:
    print(div_[:, 4])
#%%
def kmeans2(data, k = 2):
    assert len(data.shape) == 2
    n,m = data.shape
    selectP = np.array(random.sample(list(data),k)) # k*m
    tmpselectP = np.zeros([k,m])
    dist = np.zeros([n,k])
    sim = 1
    while sim >= 1e-19:
        for i in range(k):
            dist[:, i] = np.sum((selectP[i, :] - data)**2, 1)
        mindist = np.min(dist,1)
        for i in range(k):
            ids = np.where(dist[:, i] == mindist)[0]
            if len(ids) == 0: 
                tmpselectP[i, :] = random.sample(list(data), 1)[0]
            else:
                tmpselectP[i, :] = np.sum(data[ids, :], 0) / len(ids)    
        sim = np.mean(np.abs(tmpselectP - selectP))
        selectP = tmpselectP
    for i in range(k):
        dist[:, i] = np.sum((selectP[i, :] - data)**2, 1)
    mindist = np.min(dist,1)
    return selectP, (data[np.where(dist[:,i] == mindist)[0],:] for i in range(k)), (np.where(dist[:,i] == mindist)[0] for i in range(k)), np.mean(mindist)
