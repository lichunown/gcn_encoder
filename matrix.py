# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:32:14 2019

@author: lcy
"""

import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import reverse_cuthill_mckee, csgraph_from_dense, structural_rank, maximum_bipartite_matching
from scipy.sparse import csc_matrix
import seaborn as sns


def norm(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
#graph_data = np.load('data/new/result_fin.npy')
#graph_data = np.load('data/new/clustercor.npy')
#graph_data = np.load('data/new/pearsonr.npy')
#graph_data = np.load('data/new/mutual.npy')
graph_data = np.load('data/new/samplegraph_70percent.npy')
#graph_data = graph_data + graph_data.T

#graph_data[np.where(graph_data<0)] = 0
#graph_data = (graph_data-np.min(graph_data))/(np.max(graph_data)-np.min(graph_data))

#G = nx.Graph() 
#G.add_nodes_from([str(i) for i in range(len(G))])

#where = np.where(graph_data != 0)
#G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], graph_data[where]))

is_zeros = np.where(np.isnan(graph_data))
not_zeros = np.where(~np.isnan(graph_data))
max_ = np.max(graph_data[not_zeros])
min_ = np.min(graph_data[not_zeros])
graph_data = (graph_data - min_)/(max_ - min_)
graph_data[is_zeros] = np.nan


csg = csgraph_from_dense(graph_data, null_value=np.nan)
change_list = reverse_cuthill_mckee(csg, True)
#change_list = maximum_bipartite_matching(csg)
G = nx.from_scipy_sparse_matrix(csg)

show_img = nx.to_numpy_matrix(G, nodelist=change_list)
#show_img = nx.to_numpy_matrix(G)


plt.imshow(show_img, cmap='Blues')
#plt.imshow(nx.to_numpy_matrix(G, nodelist=[str(i) for i in change_list]), cmap='Blues')

#result = np.zeros(graph_data.shape)
#for i in range(424):
#    for j in range(424):
#        result[i,j] = graph_data[change_list[i], change_list[j]]
##        result[change_list[i], change_list[j]] = graph_data[i, j]
#        
#plt.imshow(result, cmap='Blues')
#%%
from sklearn.manifold import TSNE
graph_data = np.load('data/new/clustercor(2).npy') 
#graph_data = np.load('data/new/mutual.npy') 
#graph_data[np.where(graph_data < 0)] = 0


#csg = csgraph_from_dense(graph_data, null_value=0)
#change_list = reverse_cuthill_mckee(csg, True)
#G = nx.from_scipy_sparse_matrix(csg)
#graph_data = nx.to_numpy_matrix(G, nodelist=change_list)


tsne = TSNE(n_components=2)
fit_result = tsne.fit_transform(graph_data)

plt.scatter(*fit_result[0:50].T, marker='o', c=(1, 0, 0,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
plt.scatter(*fit_result[50:90].T, marker='o', c=(0, 1, 0,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
plt.scatter(*fit_result[90:278].T, marker='o', c=(0, 0, 1,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
plt.scatter(*fit_result[278:424].T, marker='o', c=(0, 0, 0,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
