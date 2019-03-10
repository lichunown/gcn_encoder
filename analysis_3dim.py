import numpy as np
import matplotlib.pyplot as plt
from data_process import yield_data_time
from mpl_toolkits.mplot3d import Axes3D


encode3dim = np.load('result/rnn_new/encode3_seqlen/finfull_e78_encode.npy')

origin_data = np.load('data/new/data2.npy')
label = np.load('data/new/label2.npy')
label = label.T

seq_label = []
for seq in yield_data_time(label, 1, 10, False):
    seq_label.append(np.sum(seq))
seq_label = np.array(seq_label)
seq_label = (seq_label>50).astype('int')

#%%

select_nums = 1000
select_ids = np.random.choice((range(len(seq_label))), select_nums)

select_label = seq_label[select_ids]
select_encode = encode3dim[select_ids]

positive_select = select_encode[np.where(select_label == 0)]
negative_select = select_encode[np.where(select_label == 1)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*positive_select.T, marker='o', c='', edgecolors='b')
ax.scatter(*negative_select.T, marker='^', c='', edgecolors='r')
plt.show()
