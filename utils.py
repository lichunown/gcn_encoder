import numpy as np
import json
import networkx as nx


def read_data():
    data_path = 'data/state.npy'
    weight_path = 'data/weight.json'
    
    n_start = 30
    n_end   = 53
    
    data = np.load(data_path)
    with open(weight_path, 'r') as f:
        weight = json.load(f)
    
    weight_values = np.array(list(weight.values()))
    weight_values = (weight_values - np.min(weight_values))/(np.max(weight_values) - np.min(weight_values))
    for i, name in enumerate(weight):
        weight[name] = weight_values[i]
    
    edges = [(*e.split('-'), weight[e]) for e in weight]
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(n_start, n_end+1)])
    G.add_weighted_edges_from(edges)
    G.add_weighted_edges_from([(str(i), str(i), 1) for i in range(n_start, n_end + 1)])
    
    A = nx.adjacency_matrix(G).todense()
    Ds = np.eye(n_end - n_start + 1)
    Ds[list(range(n_end - n_start + 1)),list(range(n_end - n_start + 1))] = np.power(np.sum(A, 0), -1/2)
#    L = D - A
    L = np.dot(np.dot(Ds, A), Ds)
    # N = D^{-1/2} L D^{-1/2}
    lplc = nx.normalized_laplacian_matrix(G).todense()
    
    data = data[:, n_start:n_end + 1]
    std = np.std(data, 0)
    mean = np.mean(data, 0)
#    data = (data - mean) / (3*std)
    data = (data - np.min(data))/(np.max(data)-np.min(data))
    return G, L, lplc, data, std, mean


G, L, lplc, data, std, mean = read_data()


def yield_data(data, batch_size, random=True):
    data = data.copy()
    max_len = len(data)
    if random:
        np.random.shuffle(data)
    i = 0
    while i + batch_size <= max_len:
        yield data[i: i + batch_size]
        i += batch_size
    yield data[i:]