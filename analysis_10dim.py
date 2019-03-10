import numpy as np
import matplotlib.pyplot as plt
from data_process import yield_data_time

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

#encode10dim = np.load('result/rnn_new/encode10_seqlen/fin_delnegative/fin_delnegative_e77_encode.npy')

origin_data = np.load('data/new/data2.npy')
label = np.load('data/new/label2.npy')
label = label.T

seq_label = []
for seq in yield_data_time(label, 1, 10, False):
    seq_label.append(np.sum(seq))
seq_label = np.array(seq_label)
seq_label = (seq_label>50).astype('int')

#%%

tsne = TSNE(n_components=2)
fit2dim = tsne.fit_transform(encode10dim)
np.save('result/rnn_new/encode10_seqlen/fin_delnegative/tsne_fit2dim.npy', fit2dim)

tsne = TSNE(n_components=3)
fit3dim = tsne.fit_transform(encode10dim)
np.save('result/rnn_new/encode10_seqlen/fin_delnegative/tsne_fit3dim.npy', fit3dim)

#%%

select_nums = 20000
select_ids = np.random.choice((range(len(seq_label))), select_nums)

select_label = seq_label[select_ids]
select_encode = fit2dim[select_ids]

positive_select = select_encode[np.where(select_label == 0)]
negative_select = select_encode[np.where(select_label == 1)]
plt.scatter(*positive_select.T, marker='o', c='', edgecolors='b')
plt.scatter(*negative_select.T, marker='^', c='', edgecolors='r')

#%%

select_nums = 5000
select_ids = np.random.choice((range(len(seq_label))), select_nums)

select_label = seq_label[select_ids]
select_encode = fit3dim[select_ids]

positive_select = select_encode[np.where(select_label == 0)]
negative_select = select_encode[np.where(select_label == 1)]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*positive_select.T, marker='o', c='', edgecolors='b')
ax.scatter(*negative_select.T, marker='^', c='', edgecolors='r')
plt.show()

#%%
 # 424 TSNE
tsne = TSNE(n_components=2)
origin_data = origin_data.T
fit424_2dim = tsne.fit_transform(origin_data)
np.save('result/rnn_new/tsne_424fit2dim.npy', fit424_2dim)

positive_select = select_encode[np.where(select_label == 0)]
negative_select = select_encode[np.where(select_label == 1)]
plt.scatter(*positive_select.T, marker='o', c='', edgecolors='b')
plt.scatter(*negative_select.T, marker='^', c='', edgecolors='r')