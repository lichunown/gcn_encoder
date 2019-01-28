import torch, os
import numpy as np

from utils import lplc24, data24, yield_data, L24, D24, yield_data_n, data_artificial, lplc_artificial
from model import Encoder, Decoder, Trainer1, Trainer3, Trainer2
from torch.optim import Adam
from gcn import GraphConvolution
import torch.nn as nn

class RMSE():
    def __call__(self, x, y):
        return torch.mean(torch.sqrt(torch.sum(torch.sum((x-y)**2, -1), -1)))

class MSE():
    def __call__(self, x, y):
        return torch.mean(torch.sum(torch.sum((x-y)**2, -1), -1))

class MAPE():
    def __call__(self, x, y):
        return torch.mean(torch.sum(torch.sum((torch.log(x.add(1)) - torch.log(y.add(1)))**2, -1), -1))
    
loss_fn = MAPE()#torch.nn.MSELoss(reduction='mean')

othername = 'boiler'

data = data24.copy()
lplc = lplc24.copy() 

#data = data_artificial.copy()
#lplc = lplc_artificial.copy()

np.random.shuffle(data)
train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):]

def normalize(data):
    min_ = np.min(data)
    max_ = np.max(data)
    return (data - max_)/(max_ - min_)

batch_size = 512
epochs = 40
dropout = 0

yield_n = 1

device = torch.device("cuda")

#encoder = Encoder().to(device)
#decoder = Decoder().to(device)
trainer = Trainer3(in_features=yield_n, in_dim=24, out_features=yield_n*8, out_dim=2, dropout=dropout).to(device)
trainer_optim = Adam(trainer.parameters(), 0.01)
#decoder_optim = Adam(decoder.parameters(), 0.001)


def test():
    trainer.eval()
    all_loss = []
    for i, xs in enumerate(yield_data_n(val_data, batch_size, yield_n)):
        xs = torch.FloatTensor(xs).to(device)
        invadj = None
        out = trainer(xs, adj, invadj)
        
#        loss = torch.mean(torch.sqrt(torch.sum(torch.sum((xs-out)**2, -1), -1)))
        loss = loss_fn(xs, out)
        all_loss.append(loss.cpu().data.numpy())
    return np.mean(all_loss)



#invadj = torch.FloatTensor(np.linalg.inv(L)).to(device)
#invadj = torch.FloatTensor(np.linalg.inv(lplc)).to(device)
#invadj = torch.FloatTensor(np.ones([24,24])).to(device)

train_loss = []
test_loss = []

p = len(train_data)//batch_size + 1 
for e in range(epochs): 
    for i, xs in enumerate(yield_data_n(train_data, batch_size, yield_n)):
        
        adj = torch.FloatTensor(lplc).to(device)
#        adj = torch.FloatTensor(np.dot(np.linalg.inv(D), lplc)).to(device)
#        invadj = torch.FloatTensor(normalize(np.linalg.inv(L))).to(device)
        invadj = None
        
        trainer.train(True)
        xs = torch.FloatTensor(xs).to(device)
        
        out = trainer(xs, adj, invadj)
        
        loss = loss_fn(xs, out)
        trainer_optim.zero_grad()
        loss.backward()
        trainer_optim.step()
        
        sloss = loss.cpu().data.numpy()
        train_loss.append((e*p+i, sloss))
        
        if i % 10 == 0:
            l = test()
            test_loss.append((e*p+i, l))
            print('e:{}/{}, i:{}/{}, loss:{}  test_loss:{}'.format(e, epochs, i,
                  len(train_data)//batch_size, sloss, l))

result = {
    'train_loss': train_loss,
    'test_loss': test_loss
}

save_dir = 'result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

encoder_layers = []
for name, layer in trainer.encoder.named_modules():
    if isinstance(layer, GraphConvolution):
        encoder_layers.append('eg{}'.format(layer.out_features))
    if isinstance(layer, nn.Linear):
        encoder_layers.append('ed{}'.format(layer.out_features))
        
decoder_layers = []
for name, layer in trainer.decoder.named_modules():
    if isinstance(layer, GraphConvolution):
        decoder_layers.append('dg{}'.format(layer.out_features))
    if isinstance(layer, nn.Linear):
        decoder_layers.append('dd{}'.format(layer.out_features))


save_name = '_'.join([str(item) for item in [othername,
                        trainer.__class__.__name__, 'epochs', epochs, 'yieldn', yield_n,
                        'batch', batch_size, 'drop', dropout, 
                        loss_fn.__class__.__name__, trainer_optim.__class__.__name__, 
                        'lr', trainer_optim.defaults['lr'], *encoder_layers, *decoder_layers
                    ]])

save_path = os.path.join(save_dir, save_name)
import pickle as pk
with open(save_path + '.pkl', 'wb') as f:
    pk.dump(result, f)
    
with open(save_path + '.model', 'wb') as f:
    torch.save(trainer, f)
    
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8), dpi=300)
plt.plot(*np.array(train_loss).T, label='train_loss')
plt.plot(*np.array(test_loss).T, label='test_loss')
plt.legend()
plt.savefig(save_path + '.png')

r = []
trainer.eval()
for i, xs in enumerate(yield_data_n(data, batch_size, yield_n, False)):
    xs = torch.FloatTensor(xs).to(device)
    adj = torch.FloatTensor(lplc).to(device)
    r.append(trainer.encoder(xs, adj).cpu().data.numpy())
result = np.concatenate(r)
np.save(save_path+'_encoding.npy', result)