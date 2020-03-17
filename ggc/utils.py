import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)

    return labels_onehot

def load_data(dataset):
    """Load feature matrix and ground truth assignment matrix."""
    print('Loading {} dataset...'.format(dataset))
    
    idx_feature_labels = np.genfromtxt("data/{}.content".format(dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_feature_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_feature_labels[:, -1])

    idx = np.array(idx_feature_labels[:, 0], dtype=np.int32)
    nodelist_mapping = dict(enumerate(idx))

    print('Samples={},Features={},Labels={}'.format(features.shape[0],features.shape[1],labels.shape[1]))

    return features.toarray(), labels, nodelist_mapping

def preprocess_features(features):
    """Row-normalize feature matrix"""
    features = preprocessing.normalize(features, norm='l1', axis=1)
    return features

def count_nodes(adj):
    """Compute number of nodes of a graph using its adjacency matrix"""
    return adj.shape[0]

def count_edges(adj):
    """Compute number of edges of a graph using its adjacency matrix"""
    return int(np.sum(np.count_nonzero(adj)) / 2)

def count_density(adj):
    """Compute density of a graph using its adjacency matrix"""
    N = adj.shape[0]
    return count_edges(adj) / (N * (N-1) / 2)

def show_info(adj):
    """Show basic information about a graph using its adjacency matrix"""
    print("Nodes={},Edges={},Density={:.5f}".format(count_nodes(adj),count_edges(adj),count_density(adj)))

def save_graph(adj, nodelist_mapping, filename):
    """Save a graph in the form of an edgelist from its adjacency matrix and node mapping"""
    G_adj = nx.relabel_nodes(nx.Graph(adj),nodelist_mapping)
    edgelists = list(nx.to_edgelist(G_adj))
    
    f = open(filename, "w")
    for i in range(len(edgelists)):
        f.write(str(edgelists[i][0]) + '\t' + str(edgelists[i][1]) + '\n')
    f.close()
