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
epochs = 70
seq_len = 10

encode_dim = 10

save_parent_dir = './result/rnn_new'
save_child_dir = 'wind1_encode{}_seqlen'.format(encode_dim, seq_len)
save_dir = os.path.join(save_parent_dir, save_child_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_name = 'wind1'
save_name = os.path.join(save_dir, save_name)

selects = np.array([3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
           24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41,
           42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 66, 67,
           72, 74, 75, 76, 79, 86, 88, 92, 94, 95, 98, 100, 104, 106, 108, 109,
           110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
           124, 125, 126, 127, 128, 130, 132, 134, 136, 137, 138, 139, 140, 141,
           142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
           156, 157, 179, 181, 182, 183, 197, 199, 206, 208, 209, 218, 219, 220,
           221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 
           236, 237, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 
           251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 280, 
           284, 301, 303, 304, 307, 308, 311, 312, 313, 314, 315, 316, 317, 318,
           319, 322, 324, 325, 326, 331, 332]) + 90

#selects = list(set(list(range(90,424))) - set(np.array([3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
#           24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41,
#           42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 66, 67,
#           72, 74, 75, 76, 79, 86, 88, 92, 94, 95, 98, 100, 104, 106, 108, 109,
#           110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
#           124, 125, 126, 127, 128, 130, 132, 134, 136, 137, 138, 139, 140, 141,
#           142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
#           156, 157, 179, 181, 182, 183, 197, 199, 206, 208, 209, 218, 219, 220,
#           221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 
#           236, 237, 238, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 
#           251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 264, 280, 
#           284, 301, 303, 304, 307, 308, 311, 312, 313, 314, 315, 316, 317, 318,
#           319, 322, 324, 325, 326, 331, 332])+90))

#selects = list(set(range(0,90))-set([4, 5, 6, 7, 8, 10, 11, 14, 17, 18, 19, 20,
#               21, 22, 23, 24, 25, 26, 28, 34, 35, 36, 37, 39, 40, 48, 49, 50, 
#               51, 53, 54, 55, 63, 64, 65, 66, 67, 68, 69, 70, 78, 79, 80, 81, 
#               82, 83, 84, 85, 88, 89]))
    
nodes_nums = len(selects)

graph_data_all = np.load('data/new/result_fin.npy')
graph_data_all = graph_data_all + graph_data_all.T

graph_data = np.zeros([len(selects), len(selects)])
for i, si in enumerate(selects):
    for j, sj in enumerate(selects):
        graph_data[i, j] = graph_data_all[si, sj]
        
        
#graph_data[np.where(graph_data < 0)] = 0

G = create_graph(graph_data)
adj = np.array(nx.normalized_laplacian_matrix(G).todense())
data = np.load('data/new/data2.npy')
info = np.load('data/new/info2.npy')
data = data.T[:, selects]
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
    
    
encoder.eval()
encoder_result = []
for i, xs_seq in enumerate(yield_data_time(data, batch_size, seq_len, False)):
    xs_seq = torch.FloatTensor(xs_seq).to(device)
    encoder_result.append(encoder(xs_seq, adj).cpu().data.numpy())
encoder_result = np.concatenate(encoder_result)
np.save(save_name+'_encode.npy', encoder_result)