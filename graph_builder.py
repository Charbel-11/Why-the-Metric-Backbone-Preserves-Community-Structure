import graphlearning as gl
import pygsp
import scipy
from pygsp import graphs

from EffectiveResistanceSampling.Network import *
from metric_backbone import *


def normalize_columns(matrix):
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms

def get_cos_weight_matrix(X, k):
    X = normalize_columns(X)
    m, N = X.shape

    Z = scipy.sparse.lil_matrix((N, N))
    for i in range(N):
        corvec = np.abs(X.T.dot(X[:, i]))  # Get scalar product of vi with all vj
        corvec[i] = 0  # Set product with itself to 0 so TSC will not select it
        order = np.argsort(corvec)[::-1]
        el = corvec[order]
        if el[0] > 1:
            el = [np.min((x, 1)) for x in el]
        Z[i, order[:k]] = np.exp(-2 * np.arccos(el[:k]))

    Z = scipy.sparse.csr_matrix(Z)
    A = (Z + Z.T) / 2
    return A


def get_Gaussian_weight_matrix(X, k):
    Z = gl.weightmatrix.knn(X, k)  # Gaussian similarity measure
    A = (Z + Z.T) / 2
    return A


def contact_to_proximity_jaccard(G_contact):
    """
    Normalizes a contact adjacency matrix using the Weighted Jaccard Coefficient so that its values are in [0,1]
    :param G_contact: Networkx graph where the weight of an edge i,j is the number of contacts between i and j
    :return: Networkx graph where the weight of an edge i,j is the proximity between i and j
    """
    N = G_contact.number_of_nodes()
    G_proximity = nx.Graph()
    G_proximity.add_nodes_from(range(N))

    neighbors = {}
    for i in range(N):
        neighbors[i] = [i]
        for j in G_contact[i]:
            neighbors[i].append(j)
        G_proximity.add_edge(i, i, proximity=1)
        G_contact.add_edge(i, i, contact=1)

    for i, j in G_contact.edges():
        if i == j:
            continue

        cur_int = set()
        cur_union = set()
        for v in neighbors[i]:
            cur_union.add(v)
            if v in neighbors[j]:
                cur_int.add(v)
        for v in neighbors[j]:
            cur_union.add(v)

        num = 0
        den = 0
        for v in cur_int:
            num += min(G_contact[i][v]['contact'], G_contact[v][j]['contact'])
        for v in cur_union:
            den += max(G_contact[i][v]['contact'] if G_contact.has_edge(i, v) else 0,
            G_contact[v][j]['contact'] if G_contact.has_edge(v, j) else 0)
        if den != 0:
            G_proximity.add_edge(i, j, proximity=num/den)

    return G_proximity


def adj_to_proximity(G):
    """
    Normalizes an adjacency matrix by the max value so that its values are in [0,1]
    :param G: Networkx graph where weight of edge i, j = number of contacts between i and j
    :return: Networkx graph normalizing weights by max
    """
    N = G.number_of_nodes()

    G_proximity = nx.Graph(G)
    max_prox = 1
    for i, j in G.edges():
        max_prox = max(max_prox, G[i][j]['weight'])
    for i, j in G_proximity.edges():
        G_proximity[i][j]["proximity"] = G_proximity[i][j].pop("weight") / max_prox
    for i in range(N):
        if not G_proximity.has_edge(i, i):
            G_proximity.add_edge(i, i, proximity=1)
    return G_proximity


def proximity_to_distance(G_proximity):
    G = nx.Graph(G_proximity)
    for i, j in G.edges():
        G[i][j]['weight'] = 1/G[i][j]['proximity']-1
    for i in range(G.number_of_nodes()):
        G.remove_edge(i, i)
    return G


def get_threshold_graph(D, num_edges):
    """
     :param D: networkx distance graph (with weight and proximity edge attribute)
     :param num_edges: the number of edges to keep
     :return: Networkx subgraph of D where we keep num_edges largest proximity edges
    """
    T = nx.Graph()
    T.add_nodes_from(D)
    edges = sorted(D.edges(data=True), key=lambda t: t[2].get('proximity', 1))
    edges.reverse()

    for i in range(num_edges):
        T.add_edge(edges[i][0], edges[i][1],
                   weight=edges[i][2]['weight'], proximity=edges[i][2]['proximity'])

    return T


def spectral_graph_sparsify(G, num_edges_to_keep):
    r"""Sparsify a graph (with Spielman-Srivastava).
    Taken from https://github.com/noamrazin/gnn_interactions/blob/master/edges_removal/spectral_sparsification.py
    Adapted from the PyGSP implementation (https://pygsp.readthedocs.io/en/v0.5.1/reference/reduction.html).

    Parameters
    ----------
    G : PyGSP graph or sparse matrix
        Graph structure or a Laplacian matrix
    num_edges_to_keep : int
        Number of edges to keep in graph.

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling` for more information.
    """
    # Test the input parameters
    if isinstance(G, graphs.Graph):
        if not G.lap_type == 'combinatorial':
            raise NotImplementedError
        L = G.L
    else:
        L = G

    N = np.shape(L)[0]

    # Not sparse
    resistance_distances = pygsp.utils.resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(G, graphs.Graph):
        W = G.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = scipy.sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = scipy.sparse.find(scipy.sparse.tril(W))

    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    results = np.random.choice(np.arange(np.shape(Pe)[0]), size=num_edges_to_keep, p=Pe, replace=False)
    new_weights = np.zeros(np.shape(weights)[0])
    new_weights[results] = 1

    sparserW = scipy.sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                 shape=(N, N))
    sparserW = sparserW + sparserW.T
    sparserL = scipy.sparse.diags(sparserW.diagonal(), 0) - sparserW

    if isinstance(G, graphs.Graph):
        sparserW = scipy.sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not G.is_directed():
            sparserW = (sparserW + sparserW.T) / 2.

        Mnew = graphs.Graph(W=sparserW)
    else:
        Mnew = scipy.sparse.lil_matrix(sparserL)

    return Mnew


def get_Spielman_graph(D, num_edges):
    n = D.number_of_nodes()

    Adj = nx.adjacency_matrix(D, [i for i in range(n)], weight='proximity')
    Adj = (Adj + Adj.T)/2
    GA = graphs.Graph(Adj)

    Gs = spectral_graph_sparsify(GA, num_edges)
    edges = Gs.get_edge_list()

    Gout = nx.Graph()
    Gout.add_nodes_from(range(n))
    for i in range(len(edges[0])):
        u, v, w = edges[0][i], edges[1][i], edges[2][i]
        Gout.add_edge(u, v, proximity=w)

    return Gout


def get_two_graphs(get_dataset, has_meta, from_contact, *args):
    """
        :param get_dataset: Function that returns the adjacency matrix of the dataset we are working with
        :param has_meta: True if we have metalabels for the dataset
        :param from_contact: True if the dataset represents a contact network
        :return: Networkx Distance graph, Networkx Metric Backbone,
    """
    if has_meta:
        G, partition_Meta = get_dataset(*args)
    else:
        G = get_dataset(*args)

    start = time.time()
    if from_contact:
        G_proximity = contact_to_proximity_jaccard(G)
    else:
        G_proximity = adj_to_proximity(G)

    D = proximity_to_distance(G_proximity)
    end = time.time()
    print("Built the distance graph in %.3f s" % (end - start))

    start = time.time()
    B = get_metric_backbone_igraph(D)
    end = time.time()
    print("Built the metric backbone graph in %.3f s" % (end - start))

    return D, B


def get_graphs(get_dataset, has_meta, from_contact, *args):
    """
        :param get_dataset: Function that returns the adjacency matrix of the dataset we are working with
        :param has_meta: True if we have metalabels for the dataset
        :param from_contact: True if the dataset represents a contact network
        :return: Networkx Distance graph, IGraph Distance Graph, Networkx Metric Backbone,
                IGraph Metric Backbone, Networkx Threshold Graph, IGraph Threshold Graph
    """
    partition_Meta = None
    if has_meta:
        G, partition_Meta = get_dataset(*args)
    else:
        G = get_dataset(*args)

    start = time.time()
    if from_contact:
        G_proximity = contact_to_proximity_jaccard(G)
    else:
        G_proximity = adj_to_proximity(G)

    D = proximity_to_distance(G_proximity)
    end = time.time()
    print("Built the distance graph in %.3f s" % (end - start))

    start = time.time()
    B = get_metric_backbone_igraph(D)
    end = time.time()
    print("Built the metric backbone graph in %.3f s" % (end - start))

    start = time.time()
    S = get_Spielman_graph(D, B.number_of_edges())
    end = time.time()
    print("Built the Spielman graph in %.3f s" % (end - start))

    start = time.time()
    T = get_threshold_graph(D, B.number_of_edges())
    end = time.time()
    print("Built the threshold graph in %.3f s" % (end - start))

    if has_meta:
        return D, partition_Meta, B, T, S
    else:
        return D, B, T, S


def approximate_Spectral_Sparsifier(G):
    network = Network(None, None, G)
    epsilon=0.1
    method='kts'
    Effective_R = network.effR(epsilon, method)
    q = 20000
    EffR_Sparse = network.spl(q, Effective_R, seed=2020)

    S = nx.Graph()
    S.add_nodes_from(range(G.number_of_nodes()))
    for idx, (u, v) in enumerate(EffR_Sparse.E_list):
        S.add_edge(u, v, proximity=EffR_Sparse.weights[idx], weight=1/EffR_Sparse.weights[idx] - 1)

    return S


def readGraphFromFile(path):
    f = open(path, 'r')
    lines = f.readlines()
    N = int(lines[0])

    B = nx.Graph()
    B.add_nodes_from(range(N))
    for i in range(1, len(lines)):
        u, v, w = lines[i].split(" ")
        u, v = int(u), int(v)
        w = float(w)
        B.add_edge(u, v, weight=w, proximity=1/(w+1))

    return B

