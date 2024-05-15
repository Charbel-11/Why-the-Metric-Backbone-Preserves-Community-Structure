import networkx as nx
import numpy as np
import scipy
from sklearn.cluster import KMeans
import igraph as ig

import time

SEED = 11


def get_partitions_Louvain(G_ig, weight='proximity'):
    start = time.time()
    cL = G_ig.community_multilevel(weights=weight)
    partition = {}
    clusterTypes = set()
    for i in range(len(cL.membership)):
        partition[i] = cL.membership[i]
        clusterTypes.add(partition[i])

    end = time.time()
    print("Louvain executed in %.3f s" % (end-start))
    return partition, len(clusterTypes)


def get_partitions_Leiden(G_ig, weight='proximity'):
    start = time.time()
    cL = G_ig.community_leiden(objective_function="modularity", weights=weight)
    partition = {}
    clusterTypes = set()
    for i in range(len(cL.membership)):
        partition[i] = cL.membership[i]
        clusterTypes.add(partition[i])

    end = time.time()
    print("Leiden executed in %.3f s" % (end-start))
    return partition, len(clusterTypes)


def get_partitions_InfoMap(G_ig, weight='proximity'):
    start = time.time()
    im = G_ig.community_infomap(edge_weights=weight)

    partition = {}
    clusterTypes = set()
    for i in range(len(im.membership)):
        partition[i] = im.membership[i]
        clusterTypes.add(partition[i])

    end = time.time()
    print("InfoMap executed in %.3f s" % (end-start))
    return partition, len(clusterTypes), im.codelength


def get_SVD_Laplacian(G, weight):
    N = G.number_of_nodes()
    L_G = np.array(nx.normalized_laplacian_matrix(G, [x for x in range(N)], weight=weight).todense())
    eig_vals, eig_vecs = scipy.linalg.eigh(L_G)
    return eig_vecs, eig_vals


def get_best_partition_SVD_kmeans(G, weight='proximity', maxK=350, fixedK=False):
    '''
    If fixedK is true, then maxK is the real value of K and we don't try another one.
    '''
    def get_partitions_SVD_kmeans(G, k, eig_vecs):
        eig_vecs = eig_vecs[:, :k]
        print("Running SVD and k-means with k=" + str(k))
        kmeans = KMeans(n_clusters=k, random_state=SEED).fit(eig_vecs)
        partition = {}
        for i in range(G.number_of_nodes()):
            partition[i] = kmeans.labels_[i]
        return partition

    start = time.time()

    K = maxK
    eig_vecs, eig_vals = get_SVD_Laplacian(G, weight)
    eig_vecs = eig_vecs.real
    if not fixedK:
        maxGap = 0
        K = 1
        for i in range(0, min(maxK, len(eig_vals) - 1)):
            if eig_vals[i+1] - eig_vals[i] > maxGap:
                maxGap = eig_vals[i+1] - eig_vals[i]
                K = i+1

    partition = get_partitions_SVD_kmeans(G, K, eig_vecs)
    end = time.time()
    print("SVD + K-Means executed in %.3f s" % (end-start))
    return partition, K


def get_best_partition_SVD_kmeans_noIsolatedVertex(G, weight='proximity', k=350, fixedK=False):
    '''
    If fixedK is true, then maxK is the real value of K and we don't try another one.
    In this implementation, we discard components of size < 5 before running spectral clustering
    '''
    start = time.time()

    thresh = 5
    nodesToRem = set()
    comps = [c for c in sorted(nx.connected_components(G), key=len)]
    for comp in comps:
        if len(comp) > thresh:
            break
        for u in comp:
            nodesToRem.add(u)

    Gc = G.copy()
    order = []
    for u in G.nodes():
        if u not in nodesToRem:
            order.append(u)
    Gc.remove_nodes_from(list(nodesToRem))

    L_G = np.array(nx.normalized_laplacian_matrix(Gc, order, weight=weight).todense())
    eig_vals, eig_vecs = scipy.linalg.eigh(L_G)
    eig_vecs = eig_vecs.real

    if not fixedK:
        maxGap = 0
        k = 1
        for i in range(0, min(500, len(eig_vals) - 1)):
            if eig_vals[i+1] - eig_vals[i] > maxGap:
                maxGap = eig_vals[i+1] - eig_vals[i]
                k = i+1

    eig_vecs = eig_vecs[:, :k]

    kmeans = KMeans(n_clusters=k, random_state=SEED).fit(eig_vecs)
    partition = {}
    idx = 0
    for i in range(G.number_of_nodes()):
        if i in nodesToRem:
            partition[i] = np.random.randint(0, k)
        else:
            partition[i] = kmeans.labels_[idx]
            idx += 1

    end = time.time()
    print("SVD + K-Means executed in %.3f s" % (end-start))
    return partition, k


def get_all_partitions(G, num_clusters=-1):
    '''
    Takes a graph with proximity edge weights and returns all the clusterings of that graph
    '''
    maxK = 500
    fixedK = False
    if num_clusters != -1:
        fixedK = True
        maxK = num_clusters

    G_ig = ig.Graph.from_networkx(G)
    partition_Louvain, clusters_Louvain = get_partitions_Louvain(G_ig, 'proximity')
    partition_Leiden, clusters_Leiden = get_partitions_Leiden(G_ig, 'proximity')
    partition_IM, clusters_IM, codeLengthD_IM = get_partitions_InfoMap(G_ig, 'proximity')
#    partition_SVD_L, clusters_SVD_L = get_best_partition_SVD_kmeans(G, 'proximity', maxK=maxK,
 #                                                                   fixedK=fixedK)
    partition_SVD_L, clusters_SVD_L = get_best_partition_SVD_kmeans_noIsolatedVertex(G, 'proximity', maxK, fixedK=fixedK)

    partitions = {}
    partitions['Louvain'] = (partition_Louvain, clusters_Louvain)
    partitions['Leiden'] = (partition_Leiden, clusters_Leiden)
    partitions['InfoMap'] = (partition_IM, clusters_IM)
    partitions['SVD_Laplacian_KMeans'] = (partition_SVD_L, clusters_SVD_L)
    return partitions
