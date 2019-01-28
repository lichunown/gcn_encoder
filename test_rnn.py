import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx

from model import Trainer, Encoder_RNN, Decoder_RNN
from data_process import create_graph, yield_data_time
from utils import MSE, normalize

import pickle as pk

batch_size = 768
seq_len = 10
nodes_nums = 450
encode_dim = 100

load_name = 'result/rnn_new/test2_with_eval'


graph_data = np.load('data/new/pearsonr.npy')
G = create_graph(graph_data[:nodes_nums, :nodes_nums])
adj = np.array(nx.normalized_laplacian_matrix(G).todense())
data = np.load('data/new/data2.npy').T[:, :nodes_nums]
data = normalize(data)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    
    
with open(load_name+'.model', 'rb') as f:
    trainer = torch.load(f).to(device)
    
encoder = trainer.encoder
decoder = trainer.decoder

adj = torch.FloatTensor(adj).to(device)
adj.requires_grad = False

result = []
max_i = (len(data)-seq_len) // batch_size
for i, x in enumerate(yield_data_time(data, batch_size, seq_len, False)):
    x = torch.FloatTensor(x).to(device)
    encode = encoder(x, adj)
    result.append(encode.cpu().data.numpy())
    print("{}/{}".format(i, max_i))
result = np.array(result)

np.save(load_name+'_encode.npy', result)

#%%
#loss_cache = pk.load()
#train_loss, eval_loss = list(zip(*loss_cache))
#train_loss = np.concatenate(train_loss)
#
#plt.plot(np.arange(0, len(train_loss), 1), train_loss)
#plt.plot(np.arange(len(train_loss)/len(eval_loss), len(train_loss)+1, len(train_loss)/len(eval_loss)), eval_loss)
