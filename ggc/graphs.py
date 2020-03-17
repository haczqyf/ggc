import itertools
import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import minimum_spanning_tree

from ggc.utils import *


def knn_graph(X, k):
    """Returns k-Nearest Neighbor (MkNN) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.
    k : int, k >= 1
        Parameter for knn: the k-th nearest neighbour.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed knn graph.
    """
    assert k < X.shape[0]

    adj_directed = kneighbors_graph(X=X,
                            n_neighbors=k,
                            p=2,
                            include_self=False,
                            ).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return adj

def mknn_graph(X, k):
    """Returns Mutual k-Nearest Neighbor (MkNN) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.
    k : int, k >= 1
        Parameter for mknn: the k-th nearest neighbour.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed mknn graph.
    """
    assert k < X.shape[0]

    adj_directed = kneighbors_graph(X=X,
                            n_neighbors=k,
                            p=2,
                            include_self=False,
                            ).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj < 2] = 0
    adj[adj >= 2] = 1
    np.fill_diagonal(adj,0)

    return adj

def cknn_graph(X, delta, k):
    """Returns Continuous k-Nearest Neighbor (CkNN) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.
    delta : float, delta > 0
        Parameter for cknn.
    k : int, k >= 1
        Parameter for cknn: the k-th nearest neighbour.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed cknn graph.
    
    References
    ----------
    .. [1] Tyrus Berry, Timothy Sauer. Consistent manifold representation for topological data analysis.
           Foundations of Data Science, 2019, 1 (1) : 1-38. doi: 10.3934/fods.2019001
    """
    assert k < X.shape[0]

    D = euclidean_distances(X, X)
    N = D.shape[0]
    np.fill_diagonal(D,0)
    D_k = np.sort(D)

    adj = np.zeros([N, N])
    adj[np.square(D) < delta * delta * np.dot(D_k[:,k].reshape(-1,1),D_k[:,k].reshape(1,-1))] = 1
    np.fill_diagonal(adj,0)

    return adj

def mst_graph(X):
    """Returns Minimum Spanning Tree (MST) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed mst graph.
    """
    D = euclidean_distances(X, X)
    adj_directed = minimum_spanning_tree(D).toarray()
    adj = adj_directed + adj_directed.T
    adj[adj > 0] = 1
    np.fill_diagonal(adj,0)

    return adj

def rmst_graph(X, gamma, k):
    """Returns Relaxed Minimum Spanning Tree (RMST) graph from the feature matrix.

    Parameters
    ----------
    X : ndarray, shape (N, F)
        N samples and F-dimensional features.
    gamma : float, gamma > 0
        Parameter for rmst.
    k : int, k >= 1
        Parameter for rmst: the k-th nearest neighbour.

    Returns
    -------
    adj : ndarray, shape (N, N)
        The adjacency matrix of the constructed rmst graph.
    
    References
    ----------
    .. [1] Beguerisse-Díaz, Mariano, Borislav Vangelov, and Mauricio Barahona.
           "Finding role communities in directed networks using role-based similarity, 
           markov stability and the relaxed minimum spanning tree."
           2013 IEEE Global Conference on Signal and Information Processing. IEEE, 2013.
    """
    D = euclidean_distances(X, X)
    N = D.shape[0]
    assert k < N
    np.fill_diagonal(D,0)

    adj = np.zeros([N, N])

    D_k = np.sort(D)
    D_k = np.tile(D_k[:,k],(N,1))
    D_k = gamma * (D_k + D_k.T)
    np.fill_diagonal(D_k,0)

    max_weight = np.zeros((N,N))
    G = nx.Graph(D)
    T = nx.minimum_spanning_tree(G)
    path = dict(nx.all_pairs_dijkstra_path(T))
    for i,j in itertools.combinations(range(N),2):
        p = path[i][j]
        path_weight = np.zeros(len(p)-1)
        for k in range(len(p)-1):
            path_weight[k] = T.edges[p[k],p[k+1]]['weight']
        max_weight[i][j] = np.amax(path_weight)
    max_weight = max_weight + max_weight.T
    np.fill_diagonal(max_weight,0)
    
    adj[D < (max_weight + D_k)] = 1
    np.fill_diagonal(adj,0)

    return adj
