import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#graph_data = np.load('data/new/result_fin.npy')
#graph_data = np.load('data/new/pearsonr.npy')
#graph_data = np.load('data/new/mutual.npy')
#graph_data = np.load('data/new/clustercor.npy')
graph_data = np.load('data/new/samplegraph_70percent.npy')

graph_data = graph_data + graph_data.T

graph_data[np.where(graph_data<0.6)] = 0

graph_data = 1 - graph_data

graph_data = np.exp(graph_data*10)
#graph_data[np.where(graph_data>0.5)] = 1
#graph_data = graph_data - np.eye(len(graph_data))

info = np.load('data/new/info2.npy')


G = nx.Graph() 
G.add_nodes_from([str(i) for i in range(424)])

where = np.where(graph_data != 0)
G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], graph_data[where]))

#G.add_weighted_edges_from([(str(i), str(i), 1) for i in range(len(G))])
#%%
#with open('data/new/info2.csv', 'w', encoding='gbk') as f:
#    for line in info:
#        f.write(','.join(line))
#        f.write('\n')

labels_name = ['磨煤机', '一次风', '二次风', '引风机', '送风机']
div_nums = [[] for _ in range(len(labels_name))]
for id_, info_ in enumerate(info):
    for i, strs in enumerate(labels_name):
        if strs in info_[4]:
            div_nums[i].append(id_)
            break
        
graph_data = 1-graph_data



colors = np.zeros(len(info))
colors[div_nums[0]] = 1
colors[div_nums[1]] = 2
colors[div_nums[2]] = 3
colors[div_nums[3]] = 4
colors[div_nums[4]] = 5
nx.draw_networkx(G, edge_color='w', node_color=colors)
#%%
def convert_to_hex(rgba_color):
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)

divs = {
#    convert_to_hex((0.9,0.1,0.8)):[np.arange(0,4)],
#    convert_to_hex((0.9,0.2,0.8)):[np.arange(15,19)],
#    convert_to_hex((0.3,0.3,0.8)):[np.arange(30,34)],
#    convert_to_hex((0.7,0.4,0.8)):[np.arange(44,48)],
#    convert_to_hex((0.4,0.5,0.9)):[np.arange(59,63)],
#    convert_to_hex((0.1,0.6,0.6)):[np.arange(74,78)],

#    convert_to_hex((1,0,0)):[np.arange(0,14)],
#    convert_to_hex((0,1,0)):[np.arange(15,29)],
#    convert_to_hex((0,0,1)):[np.arange(29,43)],
#    convert_to_hex((1,1,0)):[np.arange(43,58)],
#    convert_to_hex((1,0,1)):[np.arange(58,73)],
#    convert_to_hex((0,1,1)):[np.arange(73,88)],

     convert_to_hex((1,0,0)):[np.arange(0, 51)],
     convert_to_hex((1,0,0)):[np.arange(51, 91)],
     convert_to_hex((1,0,0)):[np.arange(91, 279)],
     convert_to_hex((1,0,0)):[np.arange(279, 424)],
#   coal
#    convert_to_hex((1,0,0)):[np.arange(0, 88)],
    
#   A
#    convert_to_hex((0,1,0)):[np.arange(95,125),np.arange(147,170), np.arange(199,225),np.arange(250,276),np.arange(309,331),np.arange(353,355),np.arange(357,378),np.arange(403,407),np.arange(411,415)],

#   引凤，送风，。。
#    convert_to_hex((1,0,1)):[np.arange(90, 191)],
#    convert_to_hex((1,1,0)):[np.arange(200, 302)],
#    convert_to_hex((0,1,0)):[np.arange(309, 398)],
#    convert_to_hex((1,0,1)):[np.arange(398, 423)],
}

#colors = []
#ids = []
#for color, id_list in divs.items():
#    ids_ = np.concatenate(id_list)
#    ids.extend(list(ids_))
#    colors.extend(len(ids_)*[color])
#ids = list(np.array(ids).astype('str'))
#colors = np.array(colors)

pos = nx.spring_layout(G, k=10, dim=2)
#pos = nx.spectral_layout(G, dim=2)
#pos = nx.nx_pydot.graphviz_layout(G)

pos = np.array(list(pos.values()))

plt.figure()


plt.scatter(*pos[:88].T, marker='o', c=(0.90588235, 0.58039216, 0.37647059,0.9), edgecolors=(0.90588235, 0.58039216, 0.37647059), linewidths=2)
plt.scatter(*pos[88:].T, marker='o', c=(0.21568627, 0.41960784, 0.42745098,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=2)

#plt.scatter(*pos[0:51].T, marker='o', c=(0, 1, 1,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
#plt.scatter(*pos[51:91].T, marker='o', c=(1, 0, 1,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
#plt.scatter(*pos[91:279].T, marker='o', c=(1, 1, 0,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)
#plt.scatter(*pos[279:424].T, marker='o', c=(0, 1, 0,0.9), edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=0)

plt.axis('off')
plt.show()
#nx.draw_networkx_nodes(G, pos, ids,  node_color=colors, c='', node_size=75)
#
#other_ids = list(np.array(list(set(range(len(G)))-set([int(i) for i in ids]))).astype('str'))
#nx.draw_networkx_nodes(G, pos, other_ids,  node_color='#cccccc', c='', node_size=75, label=other_ids, with_labels = True)

#colors = np.zeros(len(info), '<U10')
#for color, id_list in divs.items():
#    colors[np.concatenate(id_list)] = color
#colors[np.where(colors=='')] = convert_to_hex((0.8,0.8,0.8))
#
#nx.draw_networkx(G,pos, edge_color='w', node_color=colors, node_size=75, with_labels=False)
#%%

pos = nx.spectral_layout(G, dim=3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*np.array(list(pos.values()))[[int(i) for i in ids]].T, color=colors)
other_ids = list(np.array(list(set(range(len(G)))-set([int(i) for i in ids]))).astype('str'))
#nx.draw_networkx_nodes(G, pos, other_ids,  node_color='#999999', node_size=75, ax=ax)
ax.scatter(*np.array(list(pos.values()))[[int(i) for i in other_ids]].T, color='#999999')
plt.show()

#%%

pos=nx.spring_layout(G)

edge_list = []
color_list = []
width_list = []
graph_data_edge = graph_data.copy()
graph_data_edge[np.where(graph_data_edge>0.5)] = 1

for edge in G.edges():
    u, v = edge
    c = graph_data[int(u), int(v)]
    if c==1:
        continue
    edge_list.append(edge)
    color_list.append(convert_to_hex((c,c,c)))
    width_list.append(c)
nx.draw_networkx_edges(G,pos,edgelist=edge_list, edge_color=color_list, alpha=width_list, )

nx.draw_networkx_nodes(G, pos, node_color='#000000', node_size=20)