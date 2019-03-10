from model import Trainer

import torch.nn as nn
import torch.nn.functional as F

import torch, os
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import MSE, normalize
import torch.optim as optim
import numpy as np
from data_process import yield_data_time
import pickle as pk

class Denses(nn.Module):
    def __init__(self, dense_nums = []):
        super(Denses, self).__init__()
        self.dense_list = []
        for i in range(1, len(dense_nums)):
            self.dense_list.append(nn.Linear(dense_nums[i - 1], dense_nums[i]))
            
        for i, layer in enumerate(self.dense_list):
            self.add_module('dense_list_'+str(i), layer)
            
    def forward(self, x, adj=None):
        for layer in self.dense_list[:-1]:
            x = layer(x)
            x = F.relu(x, inplace=True)
        x = self.dense_list[-1](x)
        return torch.sigmoid(x)


epochs = 50
batch_size = 1024
    
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    
data = np.load('data/new/data2.npy')
data = data.T[:, 90:]
data = normalize(data)

train_data = data[:int(len(data) * 0.8)]
val_data = data[int(len(data) * 0.8):]

input_dim = data.shape[1]
encoder_list = [input_dim, 64, 10]

save_parent_dir = './result/rnn_new'
save_child_dir = 'encode_dense_'+'_'.join([str(i) for i in encoder_list])
save_dir = os.path.join(save_parent_dir, save_child_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_name = 'dense' + str(encoder_list[0])
save_name = os.path.join(save_dir, save_name)


encoder = Denses(encoder_list)
encoder_list.reverse()
decoder = Denses(encoder_list)
encoder_list.reverse()
trainer = Trainer(encoder, decoder).to(device)
optimizer = optim.Adam(trainer.parameters())
loss_fn = MSE()

loss_log = []
all_len = len(train_data)//batch_size + 1

for e in range(epochs):
    loss_ = []
    trainer.train()
    for i, x in enumerate(yield_data_time(train_data, batch_size, 1, True)):
        x = x.reshape(x.shape[1], -1)
        x = torch.FloatTensor(x).to(device)
        out = trainer(x, None)
        loss = loss_fn(x, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.append(loss.cpu().data.numpy())
        if i%10 == 0:
            print('e:{}/{}  {}/{}  loss{}'.format(e, epochs, i, all_len, loss_[-1]))
    
    trainer.eval()
    loss_eval_ = []
    for i, x in enumerate(yield_data_time(val_data, batch_size, 1, True)):
        x = x.reshape(x.shape[1], -1)
        x = torch.FloatTensor(x).to(device)
        out = trainer(x, None)
        loss = loss_fn(x, out)
        loss_eval_.append(loss.cpu().data.numpy())
    eval_loss = np.mean(loss_eval_)
    print('e:{}/{}  eval_loss:{}'.format(e, epochs, eval_loss))
    
    loss_log.append([loss_, eval_loss])

    

with open(save_name + '_e{}.model'.format(e), 'wb') as f:
    torch.save(trainer, f)


with open(save_name + 'cache.pkl', 'wb') as f:
    pk.dump(loss_log, f)
    
    
encoder.eval()
encoder_result = []
for i, x in enumerate(yield_data_time(data, batch_size, 1, False)):
    x = x.reshape(x.shape[1], -1)
    x = torch.FloatTensor(x).to(device)
    encoder_result.append(encoder(x).cpu().data.numpy())

encoder_result = np.concatenate(encoder_result)
np.save(save_name+'_encode.npy', encoder_result)
