import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import time

from model import Trainer, Encoder_RNN, Decoder_RNN
from data_process import create_graph, yield_data_time
from utils import MSE, normalize

import pickle as pk

batch_size = 256
seq_len = 10
nodes_nums = 424

# 保存的模型路径，不带.model后缀
load_name = 'result/rnn_new/vae_encode10_seqlen/VAELoss_e58'


graph_data = np.load('data/new/result_fin.npy')
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
mu_result = []
std_result = []

decoder_mu = []
decoder_std = []

start = time.time()
max_i = (len(data)-seq_len) // batch_size
for i, x in enumerate(yield_data_time(data, batch_size, seq_len, False)):
    x = torch.FloatTensor(x).to(device)
    encode, mu, std = encoder(x, adj)
    mu_output = decoder(mu, adj.transpose(1, 0))
    std_output = decoder(std, adj.transpose(1, 0))
    
    decoder_mu.append(mu_output.cpu().data.numpy())
    decoder_std.append(std_output.cpu().data.numpy())
    
    result.append(encode.cpu().data.numpy())
    mu_result.append(mu.cpu().data.numpy())
    std_result.append(std.cpu().data.numpy())
    
    print("{}/{}".format(i, max_i))
    
end = time.time()
print('time:{}'.format(end - start))

encode_z = np.concatenate(result)
encode_mu = np.concatenate(mu_result)
encode_std = np.concatenate(std_result)

decoder_mu = np.concatenate([item.transpose(1,0,2,3) for item in decoder_mu])
decoder_std = np.concatenate([item.transpose(1,0,2,3) for item in decoder_std])

input_x = np.concatenate([i.transpose(1,0,2,3) for i in yield_data_time(data, batch_size, seq_len, False)])

origin_label = np.load('data/new/label2.npy').T[:, :nodes_nums]
labels = np.concatenate([np.mean(np.mean(np.mean(i.transpose(1,0,2,3), 1), 1),1) for i in yield_data_time(origin_label, batch_size, seq_len, False)])

all_result = {
    'encode_z':encode_z,
    'encode_mu':encode_mu,
    'encode_std':encode_std,
    'decoder_mu':decoder_mu,
    'decoder_std':decoder_std,
    'input_x':input_x,
    'labels':labels,
}

save_path='./result/out.pkl'
with open(save_path, 'wb') as f:
    pk.dump(all_result, f)
    
    