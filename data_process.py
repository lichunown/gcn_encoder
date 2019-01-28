import numpy as np
import networkx as nx
import json

def normalize(data):
    min_ = np.min(data, 0)
    max_ = np.max(data, 0)
    return (data - min_)/(max_ - min_)


def load_G(path, n=6):
    with open(path, 'r') as f:
        weight = json.load(f)
    
    weight_values = np.array(list(weight.values())).reshape(-1)
    weight_values = (weight_values - np.min(weight_values))/(np.max(weight_values) - np.min(weight_values))
    for i, name in enumerate(weight):
        weight[name] = weight_values[i]
    
    edges = [(*e.split('-'), weight[e]) for e in weight]
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(n)])
    G.add_weighted_edges_from(edges)
    G.add_weighted_edges_from([(str(i), str(i), 1) for i in range(n)])
    
    A = nx.adjacency_matrix(G).todense()
    D = np.eye(n)
    D[list(range(n)),list(range(n))] = np.sum(A, 0)
    L = D - A
#    L = np.dot(np.dot(Ds, A), Ds)
    # N = D^{-1/2} L D^{-1/2}
    lplc = nx.normalized_laplacian_matrix(G).todense()
    return lplc, L, D, G


def yield_data(data, batch_size, random=True):
    data = data.copy()
    max_len = len(data)
    if random:
        np.random.shuffle(data)
    i = 0
    while i + batch_size <= max_len:
        p = data[i: i + batch_size]
        yield p.reshape(*p.shape, 1)
        i += batch_size
    p = data[i:].reshape(*data.shape, 1)
    yield p.reshape(*p.shape, 1)
    
    
def yield_data_n(data, batch_size, n, random=True):
    data = data.copy()
    max_len = len(data)
    if random:
        np.random.shuffle(data)
        
    i = 0
    while i + batch_size*n <= max_len:
        t = data[i: i + batch_size*n].reshape(batch_size, n, -1)
        yield t.transpose(0, 2, 1)
        i += batch_size*n
    b = (max_len-i)//n
    t = data[i: i + b*n]
    if list(t):
        t = t.reshape(b, n, -1)
        yield t.transpose(0, 2, 1)
    
        
def yield_data_time(data, batch_size, n, random=True):
    
    def yield_data(data, n, select):
        for i in select:
            yield data[i: i + n].reshape(n, -1, 1)
    
    select = list(range(len(data) - n))
    if random:
        np.random.shuffle(select)
        
    r = []
    for p in yield_data(data, n, select):
        r.append(p)
        if len(r)==batch_size:
            yield np.array(r).transpose(1,0,2,3)
            r = []
    if len(r) != 0:
        yield np.array(r).transpose(1,0,2,3)
        
            
def create_graph(data:np.ndarray, norm=True):
    zeros = np.where(data==0)
    if norm:
        # 下三角
        norm_data = np.tril(data, -1)
        norm_data = norm_data[np.where(norm_data!=0)]
        min_ = np.min(norm_data)
        max_ = np.max(norm_data)
        data = (data - min_)/(max_ - min_)
    data[zeros] = 0
    h, w = data.shape
    assert h == w
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(h)])
    G.add_weighted_edges_from([(str(i), str(i), 1.) for i in range(h)])
    where = np.where(data != 0)
    G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], data[where]))
#    nx.draw(g, node_size=1, width=0.03)
    return G
    