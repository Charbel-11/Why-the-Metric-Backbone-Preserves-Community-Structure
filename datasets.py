import os

import scipy
import toml
import random

import networkx as nx
import pandas as pd
import numpy as np
import pickle
import graphlearning as gl
from sklearn import datasets
from ucimlrepo import fetch_ucirepo

from graph_builder import get_graphs

initialPath = ''


# initialPath = '/content/drive/MyDrive/GSMB/'


def compressNodesAndLabels(G, idxToLabel):
    idx = 0
    nodeOrder = {}
    realIdxToLabel = {}
    for x in G.nodes():
        realIdxToLabel[idx] = idxToLabel[x]
        nodeOrder[x] = idx
        idx += 1
    G = nx.relabel_nodes(G, nodeOrder)
    return G, realIdxToLabel


def get_high_school_dataset():
    columns = ['time', 'u', 'v', 'type_u', 'type_v']
    dfEdges = pd.read_csv(initialPath + 'Datasets/high-school/High-School_data_2013.csv.gz', sep=' ', names=columns,
                          encoding='utf-8')
    edgeCounts = dfEdges.value_counts(["u", "v"])

    nodesU = set(dfEdges['u'].unique())
    nodesV = set(dfEdges['v'].unique())
    nodes = set.union(nodesU, nodesV)

    m = {}
    numNodes = 0
    for u in nodes:
        m[u] = numNodes
        numNodes += 1

    G_contact = nx.Graph()
    G_contact.add_nodes_from(range(numNodes))
    for (u, v), w in edgeCounts.items():
        if G_contact.has_edge(m[u], m[v]):
            G_contact[m[u]][m[v]]['contact'] += w
            G_contact[m[v]][m[u]]['contact'] += w
        else:
            G_contact.add_edge(m[u], m[v], contact=w)

    labels = pd.read_csv(initialPath + 'Datasets/high-school/metadata_2013.txt', sep='\t', index_col=0,
                         names=['i', 'label', 'gender'])
    idxToLabel = {}
    labelStr_to_labelInt = {}
    labelIdx = 0
    for u, row in labels.iterrows():
        if u in m:
            if row['label'] not in labelStr_to_labelInt:
                labelStr_to_labelInt[row['label']] = labelIdx
                labelIdx += 1
            idxToLabel[m[u]] = labelStr_to_labelInt[row['label']]

    return G_contact, idxToLabel


def get_primary_school_dataset():
    columns = ['time', 'u', 'v', 'type_u', 'type_v']
    dfEdges = pd.read_csv(initialPath + 'Datasets/primary-school/primaryschool.csv.gz', sep='\t', names=columns,
                          encoding='utf-8')
    edgeCounts = dfEdges.value_counts(["u", "v"])

    nodesU = set(dfEdges['u'].unique())
    nodesV = set(dfEdges['v'].unique())
    nodes = set.union(nodesU, nodesV)

    m = {}
    numNodes = 0
    for u in nodes:
        m[u] = numNodes
        numNodes += 1

    G_contact = nx.Graph()
    G_contact.add_nodes_from(range(numNodes))
    for (u, v), w in edgeCounts.items():
        if G_contact.has_edge(m[u], m[v]):
            G_contact[m[u]][m[v]]['contact'] += w
            G_contact[m[v]][m[u]]['contact'] += w
        else:
            G_contact.add_edge(m[u], m[v], contact=w)

    labels = pd.read_csv(initialPath + 'Datasets/primary-school/metadata_primaryschool.txt', sep='\t', index_col=0,
                         names=['i', 'label', 'gender'])
    idxToLabel = {}
    labelStr_to_labelInt = {}
    labelIdx = 0
    for u, row in labels.iterrows():
        if u in m:
            if row['label'] not in labelStr_to_labelInt:
                labelStr_to_labelInt[row['label']] = labelIdx
                labelIdx += 1
            idxToLabel[m[u]] = labelStr_to_labelInt[row['label']]

    return G_contact, idxToLabel


def get_USairport500_dataset():
    seen = set()
    edges = []
    with open(initialPath + 'Datasets/USairport500.txt') as f:
        lines = f.readlines()
        for line in lines:
            u, v, w = line.split(' ')
            u = int(u)
            v = int(v)
            w = int(w)
            u -= 1
            v -= 1
            seen.add(u)
            seen.add(v)
            edges.append([u, v, w])

    m = {}
    n = 0
    for u in seen:
        m[u] = n
        n += 1

    G_contact = nx.Graph()
    G_contact.add_nodes_from(range(n))
    for u, v, w in edges:
        if G_contact.has_edge(m[u], m[v]):
            G_contact[m[u]][m[v]]['contact'] += w
            G_contact[m[v]][m[u]]['contact'] += w
        else:
            G_contact.add_edge(m[u], m[v], contact=w)
    return G_contact


def get_NetSet_dataset(name):
    n = 0
    edges = []
    with open(initialPath + 'Datasets/' + name + '/adjacency.tsv') as f:
        lines = f.readlines()
        for line in lines:
            u, v, _ = line.split('\t')
            u = int(u)
            v = int(v)
            n = max(n, u + 1, v + 1)
            edges.append([u, v])

    G_contact = nx.Graph()
    G_contact.add_nodes_from(range(n))
    for u, v in edges:
        if G_contact.has_edge(u, v):
            G_contact[u][v]['contact'] += 1
            G_contact[v][u]['contact'] += 1
        else:
            G_contact.add_edge(u, v, contact=1)

    idxToLabel = {}
    with open(initialPath + 'Datasets/' + name + '/labels.tsv') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            label = int(line)
            idxToLabel[idx] = label

    Gcc = sorted(nx.connected_components(G_contact), key=len, reverse=True)
    G_contact = G_contact.subgraph(Gcc[0])

    G_contact, realIdxToLabel = compressNodesAndLabels(G_contact, idxToLabel)
    return G_contact, realIdxToLabel


def get_Political_Blogs_dataset():
    return get_NetSet_dataset("polblogs")


def get_Cite_Seer_dataset():
    return get_NetSet_dataset("citeseer")


def get_Cora_dataset():
    return get_NetSet_dataset("cora")


def get_wikiSchool_dataset():
    return get_NetSet_dataset("wikischools")


def get_OpenFlights_dataset():
    seen = set()
    edges = []
    with open(initialPath + 'Datasets/openflights.txt') as f:
        lines = f.readlines()
        for line in lines:
            u, v, w = line.split(' ')
            u = int(u)
            v = int(v)
            w = int(w)
            u -= 1
            v -= 1
            seen.add(u)
            seen.add(v)
            edges.append([u, v, w])

    m = {}
    n = 0
    for u in seen:
        m[u] = n
        n += 1

    G_contact = nx.Graph()
    G_contact.add_nodes_from(range(n))
    for u, v, w in edges:
        if G_contact.has_edge(m[u], m[v]):
            G_contact[m[u]][m[v]]['contact'] += w
            G_contact[m[v]][m[u]]['contact'] += w
        else:
            G_contact.add_edge(m[u], m[v], contact=w)
    return G_contact


def get_network_coauthor_dataset():
    G = nx.read_gml(initialPath + 'Datasets/netscience/netscience.gml')
    to_rem = []
    for i in G.nodes():
        if G.degree[i] == 0:
            to_rem.append(i)
    for i in to_rem:
        G.remove_node(i)

    idx = 0
    nodeToIdx = {}
    for i in G.nodes():
        nodeToIdx[i] = idx
        idx += 1
    G = nx.relabel_nodes(G, nodeToIdx)
    for u, v, data in G.edges(data=True):
        data["contact"] = data.pop("value")

    return G


def process_SNAP_dataset(name):
    def get_communities_without_duplicates(path, G, comp):
        file_read = open(path, "r")
        lines = file_read.readlines()

        init_num_nodes = G.number_of_nodes()
        num_nodes = G.number_of_nodes()
        clusters = []
        seen = set()
        for line in lines:
            line = line.split("\t")

            nodes = [comp[int(x)] for x in line]
            cluster = []
            for node in nodes:
                if node > init_num_nodes:
                    continue
                if node not in seen:
                    seen.add(node)
                    cluster.append(node)
                else:
                    comp[num_nodes] = num_nodes
                    G.add_node(num_nodes)
                    for edge in G.edges(node):
                        G.add_edge(num_nodes, edge[1])
                    G.add_edge(num_nodes, node)
                    cluster.append(num_nodes)
                    num_nodes += 1

            clusters.append(cluster)

        clusters = sorted(clusters, key=len, reverse=True)
        return clusters[:500]

    def read_graph(path):
        G = nx.Graph()

        file_read = open(path, "r")
        lines = file_read.readlines()

        comp = {}
        edges = []
        for line in lines:
            if "#" in line:
                continue
            edge = line.split("\t")
            u = int(edge[0])
            v = int(edge[1])
            edges.append([u, v])
            comp[u] = 1
            comp[v] = 1

        idx = 0
        for x in comp.keys():
            comp[x] = idx
            idx += 1

        for edge in edges:
            u = comp[edge[0]]
            v = comp[edge[1]]
            G.add_edge(u, v)

        return G, comp

    path = ""
    path_labels = ""
    if name == 'DBLP':
        path = initialPath + "Datasets/DBLP/com-dblp.ungraph.txt"
        path_labels = initialPath + "Datasets/DBLP/com-dblp.top5000.cmty.txt"
    elif name == 'AMAZON':
        path = initialPath + "Datasets/Amazon/com-amazon.ungraph.txt"
        path_labels = initialPath + "Datasets/Amazon/com-amazon.top5000.cmty.txt"

    G, comp = read_graph(path)
    clusters = get_communities_without_duplicates(path_labels, G, comp)

    clusterNodes = set()
    realClusters = []
    for cluster in clusters:
        G_cluster = nx.induced_subgraph(G, cluster)
        curL = nx.normalized_laplacian_matrix(G_cluster).toarray()
        lambdas, W_k = np.linalg.eigh(curL)
        if lambdas[1] < 0.1:
            continue
        for v in cluster:
            clusterNodes.add(v)
        realClusters.append(cluster)

    to_rem = []
    for i in G.nodes:
        if i not in clusterNodes:
            to_rem.append(i)
    G.remove_nodes_from(to_rem)

    for i, j in G.edges():
        G[i][j]['contact'] = 1.0
        G[j][i]['contact'] = 1.0

    realIdx = 0
    nodeToIdx = {}
    idxToLabel = {}
    for label, cluster in enumerate(realClusters):
        for v in cluster:
            idxToLabel[realIdx] = label + 1
            nodeToIdx[v] = realIdx
            realIdx += 1
    G_contact = nx.relabel_nodes(G, nodeToIdx)

    pickle.dump(G_contact, open(initialPath + "Datasets/" + name + '/graph.pickle', 'wb'))
    pickle.dump(idxToLabel, open(initialPath + "Datasets/" + name + '/idxToLabel.pickle', 'wb'))


def get_SNAP_dataset(name):
    path = initialPath + "Datasets/" + name
    G_contact = pickle.load(open(path + '/graph.pickle', 'rb'))
    idxToLabel = pickle.load(open(path + '/idxToLabel.pickle', 'rb'))
    return G_contact, idxToLabel


def get_DBLP_dataset():
    return get_SNAP_dataset("DBLP")


def get_Amazon_dataset():
    return get_SNAP_dataset("AMAZON")


def get_gl_dataset(dataset, n=-1, symmetricClusters=True, fixedSeed=False):
    if dataset == "moon":
        if n != -1:
            samples, labels = datasets.make_moons(n_samples=n, noise=0.1)
        else:
            samples, labels = datasets.make_moons(n_samples=250, noise=0.1)
        return samples, labels, 2
    else:
        if dataset == "MNIST":
            N = 70000  # Out of 70,000
            numLabels = 10  # Out of 10
        #    o_samples = gl.utils.numpy_load("Datasets/MNIST/mnist_raw.npz", 'data')
        #    o_labels = gl.utils.numpy_load("Datasets/MNIST/mnist_labels.npz", 'labels')
            o_samples, o_labels = gl.datasets.load('mnist')
        elif dataset == "FashionMNIST":
            N = 70000  # Out of 70,000
            numLabels = 10  # Out of 10
      #      o_samples = gl.utils.numpy_load("Datasets/FashionMNIST/fashionmnist_raw.npz", 'data')
      #      o_labels = gl.utils.numpy_load("Datasets/FashionMNIST/fashionmnist_labels.npz", 'labels')
            o_samples, o_labels = gl.datasets.load('fashionmnist')
        elif dataset == "Cifar":
            N = 60000  # Out of 60,000
            numLabels = 10  # Out of 10
            o_samples, o_labels = gl.datasets.load('cifar10', metric='simclr')

        if n != -1:
            N = n

        if fixedSeed:
            np.random.seed(11)
        # Keeping numLabels labels and N samples
        shuffler = np.random.permutation(len(o_samples))
        o_samples = o_samples[shuffler]
        o_labels = o_labels[shuffler]

        if symmetricClusters:
            done = set()
            usedLabels = set()
            for i in range(numLabels):
                usedLabels.add(i)
            freq = {}
            samples = []
            labels = []
            for i in range(len(o_samples)):
                if o_labels[i] in usedLabels:
                    if o_labels[i] not in freq:
                        freq[o_labels[i]] = 1
                    elif freq[o_labels[i]] > N // numLabels:
                        done.add(o_labels[i])
                        if len(done) == numLabels:
                            break
                        continue
                    else:
                        freq[o_labels[i]] += 1
                    samples.append(o_samples[i, :])
                    labels.append(o_labels[i])
            samples = np.array(samples)
            labels = np.array(labels)
        else:
            labels = o_labels[o_labels < numLabels]
            idx = 0

            samples = np.empty((len(labels), len(o_samples[0, :])))
            for i in range(len(o_samples)):
                if o_labels[i] < numLabels:
                    samples[idx] = o_samples[i, :]
                    idx += 1

            if len(samples) < N:
                N = len(samples)
            samples = samples[:N]
            labels = labels[:N]

        print("N=" + str(N) + ", " + str(numLabels) + " labels")
        return samples, labels, numLabels


def get_HAR_dataset():
    X1 = pd.read_csv('Datasets/HAR/train.csv')
    X2 = pd.read_csv('Datasets/HAR/test.csv')

    X = pd.concat([X1, X2], axis=0)
    y = pd.Categorical(X["Activity"]).codes

    X = X.drop(columns=["subject", "Activity"])
    X = X.to_numpy()

    return X, y, 6


def write_Amazon_toFile():
    D, partition_Meta, B, T, S = get_graphs(get_Amazon_dataset, True, True)
    nx.write_edgelist(D, "D_Amazon", data=("proximity",))
    nx.write_edgelist(B, "B_Amazon", data=("proximity",))
    nx.write_edgelist(T, "T_Amazon", data=("proximity",))
    nx.write_edgelist(S, "S_Amazon", data=("proximity",))

    adj, idxToLabel = get_SNAP_dataset("AMAZON")
    f = open("./amazon.txt", 'w')
    f.write(str(len(adj)) + '\n')
    for i in idxToLabel:
        f.write(str(i) + " " + str(idxToLabel[i]) + '\n')
    f.close()


def write_adj_to_file(dataset, gaussian=True):
    def normalize_columns(matrix):
        norms = np.linalg.norm(matrix, axis=0)
        return matrix / norms

    def get_weight_matrix(X, k):
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

    samples, labels, numLabels = get_gl_dataset(dataset, fixedSeed=True, symmetricClusters=False)
    N = len(samples)
    k_sqrt = int(np.sqrt(N) / 2)

    if gaussian:
        Wsqrt = gl.weightmatrix.knn(samples, k_sqrt)  # Gaussian similarity measure
    else:
        Wsqrt = get_weight_matrix(samples.T, k_sqrt)
    Wsqrt = (Wsqrt + Wsqrt.transpose()) / 2
    numEdgesKNN = Wsqrt.count_nonzero()

    path = "./" + dataset + "_" + str(k_sqrt) + ".txt"
    if not gaussian:
        path = "./" + dataset + "_DP_" + str(k_sqrt) + ".txt"
    f = open(path, 'w')
    rows, cols = Wsqrt.nonzero()
    f.write(str(N) + " " + str(numEdgesKNN) + "\n")
    for u, v in zip(rows, cols):
        f.write(str(u) + " " + str(v) + " " + str(1 / Wsqrt[u, v] - 1) + '\n')
    f.close()


if __name__ == "__main__":
    get_gl_dataset("Cifar")
    #write_adj_to_file("MNIST", False)
    #    write_adj_to_file("FashionMNIST", False)
    #    write_adj_to_file("Cifar", False)
