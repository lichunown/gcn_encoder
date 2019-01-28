import os
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
import pickle as pk

from model import Trainer, Encoder_RNN, Decoder_RNN
from data_process import create_graph, yield_data_time
from utils import MSE, normalize

batch_size = 512
epochs = 100
seq_len = 10
nodes_nums = 424
encode_dim = 10

save_name = 'result/rnn_new/test2_with_eval'


graph_data = np.load('data/new/pearsonr.npy')
G = create_graph(graph_data[:nodes_nums, :nodes_nums])
adj = np.array(nx.normalized_laplacian_matrix(G).todense())
data = np.load('data/new/data2.npy')
data = data.T[:, :nodes_nums]
data = normalize(data)

train_data = data[:int(len(data) * 0.8)]
val_data = data[int(len(data) * 0.8):]

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    
encoder = Encoder_RNN(channel_num_list=[1, 16], dense_num_list=[(16)*nodes_nums, encode_dim], nodes_nums=nodes_nums, dropout=0.).to(device)
decoder = Decoder_RNN(channel_num_list=[16, 1], dense_num_list=[encode_dim, (16)*nodes_nums], seq_len = seq_len, nodes_nums=nodes_nums, dropout=0.).to(device)
trainer = Trainer(encoder, decoder).to(device)
optimizer = optim.Adam(trainer.parameters())
loss_fn = MSE()


adj = torch.FloatTensor(adj).to(device)
adj.requires_grad = False


def evaluate(model, val_data, batch_size, seq_len):
    model.eval()
    loss_all = []
    for i, xs_seq in enumerate(yield_data_time(val_data, batch_size, seq_len)):
        xs_seq = torch.FloatTensor(xs_seq).to(device)
        out = model(xs_seq, adj)
        loss = loss_fn(xs_seq, out, [0, 1])
        loss_all.append(loss.cpu().data.numpy())
    return np.mean(loss_all)


def train_epoch(data, batch_size, seq_len, val_data = None, show_iter=None, show_forward_info='', device=device):
    loss_data = []
    max_i = (len(data)-1)//batch_size
    
    for i, xs_seq in enumerate(yield_data_time(data, batch_size, seq_len)):
        trainer.train(True)
        xs_seq = torch.FloatTensor(xs_seq).to(device)
        out = trainer(xs_seq, adj)
        loss = loss_fn(xs_seq, out, [0, 1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_data.append(loss.cpu().data.numpy())
        if show_iter is not None and i%show_iter==0:
            print('{} b:{}/{}  loss:{}'.format(show_forward_info, i, max_i, loss_data[-1]))
            
    if val_data is not None:
        del xs_seq, out, loss
        torch.cuda.empty_cache()
        trainer.eval()
        eval_loss_all = []
        for i, xs_seq in enumerate(yield_data_time(val_data, batch_size, seq_len)):
            xs_seq = torch.FloatTensor(xs_seq).to(device)
            out = trainer(xs_seq, adj)
            loss = loss_fn(xs_seq, out, [0, 1])
            eval_loss_all.append(loss.cpu().data.numpy())
        val_loss = np.mean(eval_loss_all)
        print('{}  val_loss:{}'.format(show_forward_info, val_loss))
        
    return np.array(loss_data), val_loss


results = []
for e in range(epochs):
    r = train_epoch(train_data, batch_size, seq_len, val_data=val_data, show_iter=10, 
                    show_forward_info='epoch:{}/{} '.format(e, epochs))
    results.append(r)
    with open(save_name + '_e{}.model'.format(e), 'wb') as f:
        torch.save(trainer, f)


    
with open(save_name + 'cache.pkl', 'wb') as f:
    pk.dump(results, f)