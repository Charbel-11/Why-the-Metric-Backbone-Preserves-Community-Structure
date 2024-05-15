import math
import random
import time

import networkx as nx
import numpy as np
import igraph as ig
import rustworkx


def get_metric_backbone_rustworkx(D):
    N = D.number_of_nodes()
    s = time.time()
    D_rust = rustworkx.networkx_converter(D, keep_attributes=True)
    distances = rustworkx.all_pairs_dijkstra_path_lengths(D_rust, lambda e: e['weight'])
    e = time.time()
    print("Time for APSP: %.3f s" % (e-s))

    s = time.time()
    B = nx.Graph(D)
    B.remove_edges_from([(x, y) for x, y, w in B.edges.data('weight') if w > distances[x][y]])
    e = time.time()
    print("Time for edge removal: %.3f s" % (e-s))

    return B


def get_metric_backbone_igraph(D):
    """
     :param D: networkx distance graph (with weight and proximity edge attribute)
     :return: Networkx Metric Backbone subgraph of D
    """
    D_ig = ig.Graph.from_networkx(D)
    distances = D_ig.distances(weights='weight')

    B = nx.Graph(D)
    B.remove_edges_from([(x, y) for x, y, w in B.edges.data('weight') if w > distances[x][y]])
    return B


def get_metric_backbone_igraph_slow(D):
    D_ig = ig.Graph.from_networkx(D)
    edges_to_keep = set()
#    nodes = [u for u in D.nodes()][:-1]
    nodes = [u for u in D.nodes()]
    nodes = nodes[:-1]
    for u in nodes:
        cur_shortest_paths = D_ig.get_shortest_paths(u, weights='weight')
        for path in cur_shortest_paths:
            for i in range(1, len(path)):
                edges_to_keep.add((path[i-1], path[i]))

    B = nx.Graph(D)
    B.remove_edges_from([(x, y) for x, y in B.edges() if (x, y) not in edges_to_keep])
    return B


def get_approximate_metric_backbone_igraph(D):
    D_ig = ig.Graph.from_networkx(D)

    num_samples = 2*int(math.log(D.number_of_nodes())) + 1
    num_samples = min(num_samples, D.number_of_nodes())
    options = [u for u in D.nodes()]
    random.shuffle(options)
    options = options[:num_samples]
    print("Taking %s shortest path trees" % num_samples)

    edges_to_keep = set()
    for u in options:
        cur_shortest_paths = D_ig.get_shortest_paths(u, weights='weight')
        for path in cur_shortest_paths:
            for i in range(1, len(path)):
                edges_to_keep.add((path[i-1], path[i]))
                edges_to_keep.add((path[i], path[i-1]))

    B = nx.Graph(D)
    B.remove_edges_from([(x, y) for x, y in B.edges() if (x, y) not in edges_to_keep])
    return B


def get_metric_backbone_hop_optimization(D0):
    """
     :param D0: networkx distance graph (with weight and proximity edge attribute)
     :return: Networkx Metric Backbone subgraph of D0
    """
    def identify_metric_edges(D):
        M = set()
        U = {}
        minE = {}
        for v in D.nodes:
            U[v] = sorted(D.edges(v, data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
            if len(U[v]) == 0:
                continue
            minE[v] = U[v][-1][2]['weight']

        for v in D.nodes:
            if len(U[v]) == 0:
                continue
            curM = [(U[v][-1][0], U[v][-1][1])]
            M.add((U[v][-1][0], U[v][-1][1]))
            M.add((U[v][-1][1], U[v][-1][0]))
            U[v].pop()

            while len(U[v]) > 0:
                e = U[v][-1]
                U[v].pop()
                metric = True

                for m in curM:
                    x = m[1]
                    wx = D[v][x]['weight'] + minE[x]
                    if e[2]['weight'] > wx:
                        metric = False
                        break

                if metric:
                    curM.append((e[0], e[1]))
                    M.add((e[0], e[1]))
                    M.add((e[1], e[0]))
        return M

    def remove_first_order_semi_metric_edges(D):
        """
         :param D: networkx distance graph (with weight and proximity edge attribute)
         :return: Networkx subgraph of D without first order semi-metric edges
        """
        for u in D.nodes:
            adj = [v for v in D[u]]
            for i in range(len(adj)):
                for j in range(i + 1, len(adj)):
                    x = adj[i]
                    y = adj[j]
                    if D.has_edge(x, y):
                        if D[u][x]['weight'] > D[x][y]['weight'] + D[y][u]['weight']:
                            D.remove_edge(u, x)
                            break
        return D

    D = D0.copy()
#    print("Initial number of edges:", D.number_of_edges())
    D = remove_first_order_semi_metric_edges(D)
#    print("Number of edges without first order semi-metrics:", D.number_of_edges())
    metricEdges = identify_metric_edges(D)
#    print("Number of identified metric edges:", len(metricEdges)/2)
    nonMetricEdges = set()

    D_ig = ig.Graph.from_networkx(D)

    dist = {}
    edges = D.edges(data=True)
    for e in edges:
        if (e[0], e[1]) in metricEdges or (e[1], e[0]) in metricEdges or (e[1], e[0]) in nonMetricEdges:
            continue
        if e[0] not in dist:
            dist[e[0]] = D_ig.distances(e[0], weights='weight')[0]

        if dist[e[0]][e[1]] < e[2]['weight']:
            nonMetricEdges.add((e[0], e[1]))
            nonMetricEdges.add((e[1], e[0]))
        else:
            metricEdges.add((e[0], e[1]))
            metricEdges.add((e[1], e[0]))

    for e in nonMetricEdges:
        if D.has_edge(e[0], e[1]):
            D.remove_edge(e[0], e[1])

    return D
