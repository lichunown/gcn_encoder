import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csc_matrix
graph_data = np.load('data/new/result_fin.npy')
#graph_data = np.load('data/new/pearsonr.npy')
#graph_data = np.load('data/new/mutual.npy')
graph_data = graph_data + graph_data.T

graph_data[np.where(graph_data<0)] = 0

#graph_data = 1 - graph_data

for i, line in enumerate(graph_data):
    select = np.array(sorted(zip(range(len(line)), line), key=lambda x:x[1], reverse=True)[50:])[:, 0]
    graph_data[i, select.astype('int')] = 0

graph_data = (graph_data + graph_data.T)/2
graph_data = (graph_data-np.min(graph_data))/(np.max(graph_data)-np.min(graph_data))
#graph_data = np.exp(graph_data*10)
#graph_data[np.where(graph_data>0.5)] = 1
#graph_data = graph_data - np.eye(len(graph_data))

info = np.load('data/new/info2.npy')


G = nx.Graph() 
G.add_nodes_from([str(i) for i in range(len(G))])

where = np.where(graph_data != 0)
G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], graph_data[where]))

#%%

def specturalCluster(X, clusterNumber):
    adjacentMatrix = X
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    Laplacian =  np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

    lam, H = np.linalg.eig(Laplacian)

    sp_kmeans = KMeans(n_clusters=clusterNumber).fit(H)
    return sp_kmeans.labels_

clusterNumber = 3
min_result = specturalCluster(graph_data, clusterNumber)

div = [[] for _ in range(clusterNumber)]
for i, info_ in enumerate(info):
    div[min_result[i]].append(info_)
    
for i, div_ in enumerate(div):
    div[i] = np.array(div_)

for div_ in div:
    print(div_[:, 4])