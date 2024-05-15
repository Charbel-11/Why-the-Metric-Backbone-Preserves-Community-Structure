
import sklearn
from infomap import Infomap
from matplotlib import pyplot as plt

from datasets import *


def getModularity(G, partition, weight='proximity'):
    """
    :param G: networkx graph
    :param partition: dictionary mapping node index to cluster id
    :param weight: weight attribute used for proximity in G
    :return: Modularity measure
    """
    n = G.number_of_nodes()
    A = np.array(nx.adjacency_matrix(G, [x for x in range(n)], weight=weight).todense())

    m = np.sum(A) / 2
    k = np.sum(A, axis=1)
    modularity = 0

    for i in range(n):
        for j in range(n):
            if G.has_edge(i, j):
                assert(A[i][j] == G[i][j][weight])
            else:
                assert(A[i][j] == 0)
            if partition[i] == partition[j]:
                modularity += (A[i][j] - k[i] * k[j] / (2 * m))

    return modularity / (2 * m)


def getModularity2(G, partition, weight='proximity'):
    """
    This is an alternate way of finding modularity
    :param G: networkx graph
    :param partition: dictionary mapping node index to cluster id
    :param weight: weight attribute used for proximity in G
    :return: Modularity measure
    """
    A = nx.to_numpy_array(G, weight=weight)
    n = A.shape[0]
    m = np.sum(A) / 2

    k = np.sum(A, axis=1)
    K = max(partition.values()) + 1

    e = np.zeros(K)
    a = np.zeros(K)
    for i in range(n):
        a[partition[i]] += k[i]

    for i in range(n):
        for j in range(n):
            assert(A[i][j] == A[j][i])
            if partition[i] == partition[j]:
                e[partition[i]] += A[i][j]

    a2 = np.zeros(K)
    for node in G.nodes():
        a2[partition[node]] += G.degree(node, weight=weight)

    modularity = 0
    for i in range(K):
#        print(i, a[i], a2[i])
        modularity += e[i]/(2*m) - ((a[i]/(2*m))**2)
    return modularity


def getCodeLength(G, partition, weight='proximity'):
    with open('temp.clu', 'w', encoding="utf-8") as f:
        for i in range(G.number_of_nodes()):
            f.write(str(i) + " " + str(partition[i] + 1) + '\n')
    im = Infomap(silent=True, no_infomap=True, cluster_data='temp.clu')
    im.add_networkx_graph(G, weight=weight)
    im.run()
    os.remove('temp.clu')
    return im.codelength


def get_partitions_similarity_jaccard(p1, p2):
    """
    :param p1: A partition of a vertex set
    :param p2: Another partition of the same vertex set
    :return: Measure of bidirectional similarity of modularity
    """
    if p1 is None or p2 is None:
        return -1

    def jaccard(s1, s2):
        intersection = len(list(set(s1).intersection(s2)))
        union = (len(s1) + len(s2)) - intersection
        return float(intersection) / union

    clusters1 = {}
    clusters2 = {}
    for u in p1.keys():
        if p1[u] in clusters1:
            clusters1[p1[u]].add(u)
        else:
            clusters1[p1[u]] = {u}
    for u in p2.keys():
        if p2[u] in clusters2:
            clusters2[p2[u]].add(u)
        else:
            clusters2[p2[u]] = {u}

    m1 = len(clusters1)
    m2 = len(clusters2)

    similarity = 0
    for c1 in clusters1:
        for c2 in clusters2:
            similarity += jaccard(clusters1[c1], clusters2[c2])
    similarity /= np.sqrt(m1 * m2)

    return similarity


def get_partitions_similarity_NMI(p1, p2):
    if p1 is None or p2 is None:
        return -1

    vals1 = []
    vals2 = []
    for i in p1.keys():
        vals1.append(p1[i])
        vals2.append(p2[i])
    return sklearn.metrics.normalized_mutual_info_score(vals1, vals2)


def get_partitions_similarity_ARI(p1, p2):
    if p1 is None or p2 is None:
        return -1

    vals1 = []
    vals2 = []
    for i in p1.keys():
        vals1.append(p1[i])
        vals2.append(p2[i])
    return sklearn.metrics.adjusted_rand_score(vals1, vals2)


def get_partitions_similarities(partitions_A, partitions_B):
    similarities = []
    for algo in partitions_A:
        (partitionA, clusterA) = partitions_A[algo]
        (partitionB, clusterB) = partitions_B[algo]
        similarities.append(get_partitions_similarity_ARI(partitionA, partitionB))
    return similarities


def get_degree_distribution(G, title):
    degrees = [degree for node, degree in G.degree()]

    fig, axes = plt.subplots()
    axes.hist(degrees, bins=100)

    plt.savefig("Results/" + title + "_unweighted_degree_distribution.jpg")


def get_incluster_degree_distribution(G, partition, title):
    degreeOf = {}
    for u in G.nodes():
        degreeOf[u] = 0

    for u, v, d in G.edges(data=True):
        if partition[u] != partition[v]:
            continue
        degreeOf[u] += d['proximity']

    degrees = degreeOf.values()

    fig, axes = plt.subplots()
    axes.set_title('Incluster Proximity Degree Distribution')
    axes.hist(degrees, bins=100)
    plt.savefig("Results/" + title + "_proximity_incluster_degree_distribution.jpg")


def get_outcluster_degree_distribution(G, partition, title):
    degreeOf = {}
    for u in G.nodes():
        degreeOf[u] = 0

    for u, v, d in G.edges(data=True):
        if partition[u] == partition[v]:
            continue
        degreeOf[u] += d['proximity']

    degrees = degreeOf.values()

    fig, axes = plt.subplots()
    axes.set_title('Outcluster Proximity Degree Distribution')
    axes.hist(degrees, bins=100)
    plt.savefig("Results/" + title + "_proximity_outcluster_degree_distribution.jpg")


def get_cluster_size_distribution(partition, title):
    sizes = {}
    for u in partition:
        p = partition[u]
        if p not in sizes:
            sizes[p] = 1
        else:
            sizes[p] += 1

    fig, axes = plt.subplots()
    axes.set_title('Leiden Cluster Size Distribution')
    axes.hist(sizes.values())
    plt.savefig("Results/" + title + "_clusters_size_distribution.jpg")


def check_backbone_removed_edge_distribution(D, B, T, title):
    removed_proximity_B = []
    removed_proximity_T = []
    for u, v, d in D.edges(data=True):
        if not B.has_edge(u, v):
            removed_proximity_B.append(d['proximity'])
        if not T.has_edge(u, v):
            removed_proximity_T.append(d['proximity'])

    if len(removed_proximity_T) == 0:
        return

    max_range = max(np.max(removed_proximity_B), np.max(removed_proximity_T))
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[1].hist(removed_proximity_B, bins=100, range=(0, max_range))
    axs[0].hist(removed_proximity_T, bins=100, range=(0, max_range))
    axs[1].set_title("Metric Backbone")
    axs[0].set_title("Threshold Graph")

    # For common x/y label
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Proximity")
    plt.ylabel("Number of removed edges")

    plt.savefig("Results/" + title + "_removed_edges.jpg")

