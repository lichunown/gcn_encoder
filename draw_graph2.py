import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def convert_to_hex(rgba_color):
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)

#graph_data = np.load('data/new/result_fin.npy')
#graph_data = np.load('data/new/pearsonr.npy')
#graph_data = np.load('data/new/mutual.npy')
graph_data = np.load('data/new/clustercor.npy')
info = np.load('data/new/info2.npy')
#graph_data = np.load('data/new/samplegraph_70percent.npy')

is_zeros = np.where(graph_data == 0)
not_zeros = np.where(graph_data != 0)
max_ = np.max(graph_data[not_zeros])
min_ = np.min(graph_data[not_zeros])
graph_data = (graph_data - min_)/(max_ - min_)
graph_data[is_zeros] = 0

predict_div = {}
predict_div['coal_zc'] = [4, 5, 6, 7, 8, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23, 24, 
                         25, 26, 28, 34, 35, 36, 37, 39, 40, 48, 49, 50, 51, 53, 54, 
                         55, 63, 64, 65, 66, 67, 68, 69, 70, 78, 79, 80, 81, 82, 83,
                         84, 85, 88, 89]
predict_div['coal_wd'] = list(set(range(90)) - set(predict_div['coal_zc']))
predict_div['wind_1'] = np.array([3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37,
           38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
           55, 57, 66, 67, 72, 74, 75, 76, 79, 86, 88, 92, 94, 95, 98, 100, 
           104, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
           119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 130, 132, 134,
           136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 
           149, 150, 151, 152, 153, 154, 155, 156, 157, 179, 181, 182, 183,
           197, 199, 206, 208, 209, 218, 219, 220, 221, 222, 223, 224, 225, 
           226, 227, 228, 229, 230, 231, 232, 233, 235, 236, 237, 238, 240, 
           241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 
           254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 280, 284, 301, 
           303, 304, 307, 308, 311, 312, 313, 314, 315, 316, 317, 318, 319, 
                   322, 324, 325, 326, 331, 332]) + 90
predict_div['wind_2'] = list(set(range(90, 424)) - set(list(predict_div['wind_1'])))


true_div = {
    'coal_zc':[],
    'coal_wd':[],
    'wind_1':[],
    'wind_2':[],
    'None':[]
}
for i in  range(0, 90):
    if '轴承' in info[i][4]:
        true_div['coal_zc'].append(i)
    elif '温度' in info[i][4]:
        true_div['coal_wd'].append(i)
    else:
        true_div['None'].append(i)

for i in  range(90, 424):
    if 'U1' in info[i][2]:
        true_div['wind_1'].append(i)
    elif 'U2' in info[i][2]:
        true_div['wind_2'].append(i)

for i in  range(90, 424):
    if '1A' in info[i][4] or '1B' in info[i][4] or '1号' in info[i][4]:
        true_div['wind_1'].append(i)
    elif '2A' in info[i][4] or '2B' in info[i][4] or '2号' in info[i][4]:
        true_div['wind_2'].append(i)
        
#not_div = set(range(424)) - set(np.concatenate(list(true_div.values())))

colors_div = {
    'coal_zc':convert_to_hex(np.array([250, 214, 137])/255),
    'coal_wd':convert_to_hex(np.array([123, 144, 210])/255),
    'wind_1': convert_to_hex(np.array([106, 131, 114])/255),
    'wind_2': convert_to_hex(np.array([199, 62, 58])/255),
    'None': convert_to_hex((1, 1, 1)),
}

marker_div = {
    'coal_zc':'o',
    'coal_wd':'^',
    'wind_1': '*',
    'wind_2': 'p',
    'None': '.',
}

colors_list = []
marker_list = []
for i in range(424):
    for name in predict_div:
        if i in predict_div[name]:
            colors_list.append(colors_div[name])
    for name in true_div:
        if i in true_div[name]:
            marker_list.append(marker_div[name])
            
#%%
wdge_weight = np.load('data/new/90percent.npy')
wdge_weight[np.where(np.isnan(wdge_weight))] = 0
lines = []
for i in range(len(wdge_weight)):
    for j in range(i+1, len(wdge_weight)):
        if not np.isnan(wdge_weight[i,j]):
            if wdge_weight[i,j] > 1.5:
                lines.append((i, j))


#%%
graph_data = np.load('data/new/result_fin.npy')


graph_data[np.where(graph_data<0)] = 0
graph_data = 1 - graph_data
graph_data = np.exp(graph_data*10)

G = nx.Graph() 
G.add_nodes_from([str(i) for i in range(424)])
where = np.where(graph_data != 0)
G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], graph_data[where]))
#G.add_nodes_from([str(i) for i in range(len(G))])


pos = nx.spring_layout(G, k=38, dim=2, iterations=50)
pos = np.array(list(pos.values()))

plt.figure()

# draw edge
for i,j in lines:
    plt.plot(*zip(pos[i], pos[j]), color='#555555', linewidth=0.5, alpha=0.1, zorder=20)
             
# draw node
for i in range(424):
    if marker_list[i] == '.' or colors_list[i]==convert_to_hex((1, 1, 1)):
        continue
    plt.scatter(*pos[i].T, marker=marker_list[i], c=colors_list[i], edgecolors=colors_list[i], linewidths=1, zorder=30)

plt.axis('off')


#%%
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

color_patch = [
    mpatches.Patch(color=list(colors_div.values())[0], label='Predict Bearing Temperature'),
    mpatches.Patch(color=list(colors_div.values())[1], label='Predict Pulverized Coal Temperature'),
    mpatches.Patch(color=list(colors_div.values())[2], label='Predict Blower-unit 1'),
    mpatches.Patch(color=list(colors_div.values())[3], label='Predict Blower-unit 2'),
]

l1 = plt.legend(handles=color_patch, loc='upper right', prop = {'size':8})#.set_zorder(40)


now_axis = plt.axis()

p1 = plt.scatter([100],[100], marker=list(marker_div.values())[0], color='#000000')
p2 = plt.scatter([100],[100], marker=list(marker_div.values())[1], color='#000000')
p3 = plt.scatter([100],[100], marker=list(marker_div.values())[2], color='#000000')
p4 = plt.scatter([100],[100], marker=list(marker_div.values())[3], color='#000000')
                 
plt.legend([p1, p2, p3, p4], ['Original Bearing Temperature', 
           'Original Pulverized Coal Temperature', 'Original Blower-unit 1',
           'Original Blower-unit 2'], loc='upper left', scatterpoints=1, prop = {'size':8})#.set_zorder(40)
plt.gca().add_artist(l1)
plt.axis(np.array(now_axis) + [0,0,0,0.3])

#import matplotlib.collections as mcoll
#lc = mcoll.LineCollection(lines, array=z, cmap=cmap, norm=norm,
#                              linewidth=linewidth, alpha=alpha)
#ax = plt.gca()
#ax.add_collection(lc)

#plt.scatter(*pos[:88].T, marker='o', c='', edgecolors=(0.90588235, 0.58039216, 0.37647059), linewidths=2)
#plt.scatter(*pos[88:].T, marker='o', c='', edgecolors=(0.21568627, 0.41960784, 0.42745098), linewidths=2)



