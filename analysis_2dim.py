import numpy as np
import matplotlib.pyplot as plt
from data_process import yield_data_time

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

encode2dim = np.load('result/rnn_new/encode2_seqlen/finfull_e47_encode.npy')
#encode2dim = np.concatenate(encode2dim)

origin_data = np.load('data/new/data2.npy')
label = np.load('data/new/label2.npy')
label = label.T


seq_label = []
for seq in yield_data_time(label, 1, 10, False):
    seq_label.append(np.sum(seq))
seq_label = np.array(seq_label)
seq_label = (seq_label>50).astype('int')#/np.max(seq_label)

#np.save('data/new/seq_label.npy', seq_label)
#%% show encoding figure

#show_dpi = 500
#show_fig = np.zeros([show_dpi, show_dpi, 3])
#x, y = encode2dim.T * show_dpi
#
#show_fig[x.astype('int'), y.astype('int')] = [[1, 1-item, 1-item] for item in seq_label]
#plt.imshow(show_fig)

# %% draw plt figure (random select)

select_nums = 1000
select_ids = np.random.choice((range(len(seq_label))), select_nums)

select_label = seq_label[select_ids]
select_encode = encode2dim[select_ids]

positive_select = select_encode[np.where(select_label == 0)]
negative_select = select_encode[np.where(select_label == 1)]
plt.scatter(*positive_select.T, marker='o', c='', edgecolors='b')
plt.scatter(*negative_select.T, marker='^', c='', edgecolors='r')

#%% 

def isolationForest(data):
    ilf = IsolationForest(n_estimators=100,n_jobs=-1,verbose=2,contamination=0.1)
    ilf.fit(data)
    pred = ilf.predict(data)
    return pred

def LOF(data):
    lof = LocalOutlierFactor(contamination=0.01)
    pred = lof.fit_predict(data)
    return pred

def TPR(true_label, predict_label, true_div=(0, 1), predict_div=(1, -1)):
    predict_label = predict_label[5:-5]
    true_positive = np.where(true_label == true_div[0])[0]
    predict_positive = np.where(predict_label == predict_div[0])[0]
    succeed_ = set(true_positive) & set(predict_positive)
    return len(succeed_)/len(true_positive)


def FPR(true_label, predict_label, true_div=(0, 1), predict_div = (1, -1)):
    predict_label = predict_label[5:-5]
    true_negative = np.where(true_label == true_div[1])[0]
    predict_positive = np.where(predict_label == predict_div[0])[0]
    succeed_ = set(true_negative) & set(predict_positive)
    return len(succeed_)/len(true_negative)

def TNR(true_label, predict_label, true_div=(0, 1), predict_div = (1, -1)):
    predict_label = predict_label[5:-5]
    true_negative = np.where(true_label == true_div[1])[0]
    predict_negative = np.where(predict_label == predict_div[1])[0]
    succeed_ = set(true_negative) & set(predict_negative)
    return len(succeed_)/len(true_negative)


#isolation_predict = isolationForest(origin_data.T)[5:-5]
#LOF_predict = LOF(origin_data.T)[5:-5]
#print( TPR(seq_label, isolation_predict))


