from tabulate import tabulate

from metrics import *
from datasets import *
from clustering import *
from graph_builder import *


def build_tables_clustering(D, B, T, title):
    rows = ['Distance Graph', 'Metric Backbone', 'Threshold Graph']
    numTriangles = []
    numTransitivity = []
    averageClustering = []
    for G in [D, B, T]:
        curTriangles = nx.triangles(G)
        totalTriangles = sum(curTriangles.values()) / 3
        numTriangles.append(totalTriangles)
        numTransitivity.append(nx.transitivity(G))
        averageClustering.append(nx.average_clustering(G, weight='proximity'))

    clustering_table = tabulate({'Triangles': numTriangles,
                                 'Transitivity': numTransitivity,
                                 'Average Clustering': averageClustering},
                                headers='keys',
                                tablefmt='fancy_grid',
                                missingval='-',
                                showindex=rows,
                                floatfmt=".3f")

    with open('Results/' + title + '_clustering.txt', 'w', encoding="utf-8") as f:
        f.write(clustering_table)


def get_num_edges_table(D, B, S, numClusters=-1):
    numEdgesD = D.number_of_edges()
    numEdgesB = B.number_of_edges()
    numEdgesS = S.number_of_edges()
    edgesRatioMB = numEdgesB / numEdgesD
    edgesRatioSS = numEdgesS / numEdgesD

    graph_description_table = tabulate({'Number of Nodes': [D.number_of_nodes()],
                                        'Number of Edges in Distance Graph': [numEdgesD],
                                        'Number of Edges in Metric Backbone': [numEdgesB],
                                        'Number of Edges in Spectral Sparsifier': [numEdgesS],
                                        'edgesRatioMB': [edgesRatioMB],
                                        'edgesRatioSS': [edgesRatioSS],
                                        'Number of Clusters': [numClusters]},
                                       headers='keys',
                                       tablefmt='fancy_grid',
                                       missingval='-',
                                       floatfmt=".3f")

    return graph_description_table


def get_original_partition_table(D, partitionsD, partitionTypes, partition_Meta=None, clusters_Meta=-1):
    numClustersD = []
    modularitiesD = []
    similarity_metaLabels_D = []
    if partition_Meta is not None:
        modularitiesD.append(getModularity(D, partition_Meta, weight='proximity'))
        similarity_metaLabels_D.append(1)
        numClustersD.append(clusters_Meta)
    for algo in partitionsD:
        (partition, cluster) = partitionsD[algo]
        numClustersD.append(cluster)
        if partition is None:
            modularitiesD.append(-1)
            similarity_metaLabels_D.append(-1)
        else:
            modularitiesD.append(getModularity(D, partition, weight='proximity'))
            similarity_metaLabels_D.append(get_partitions_similarity_ARI(partition_Meta, partition))

    partition_table_D = tabulate({'Number of Clusters': numClustersD,
                                  'Modularity': modularitiesD,
                                  'Similarity with MetaLabels': similarity_metaLabels_D},
                                 headers='keys',
                                 tablefmt='fancy_grid',
                                 missingval='-',
                                 showindex=partitionTypes,
                                 floatfmt=".3f")

    return partition_table_D


def get_comparison_partition_table(G, partitions, partitionTypes, partitions_D, partition_Meta=None, clusters_Meta=-1):
    numClusters = []
    modularities = []
    similarity_D = []
    similarity_metaLabels = []
    if partition_Meta is not None:
        modularities.append(getModularity(G, partition_Meta, weight='proximity'))
        similarity_D.append(1)
        similarity_metaLabels.append(1)
        numClusters.append(clusters_Meta)

    for algo in partitions:
        (partition, cluster) = partitions[algo]
        (partitionD, clusterD) = partitions_D[algo]
        numClusters.append(cluster)
        if partition is None or partitionD is None:
            modularities.append(-1)
            similarity_D.append(-1)
            similarity_metaLabels.append(-1)
        else:
            modularities.append(getModularity(G, partition, weight='proximity'))
            similarity_D.append(get_partitions_similarity_ARI(partitionD, partition))
            similarity_metaLabels.append(get_partitions_similarity_ARI(partition_Meta, partition))

    partition_table = tabulate({'Number of Clusters': numClusters,
                                'Modularity': modularities,
                                'Similarity with MetaLabels': similarity_metaLabels,
                                'Similarity with Distance Graph': similarity_D},
                               headers='keys',
                               tablefmt='fancy_grid',
                               missingval='-',
                               showindex=partitionTypes,
                               floatfmt=".3f")

    return partition_table


def compute_graphs_statistics():
    get_dataset_array = [get_high_school_dataset, get_primary_school_dataset, get_USairport500_dataset,
                         get_network_coauthor_dataset, get_OpenFlights_dataset, get_DBLP_dataset,
                         get_Amazon_dataset, get_Political_Blogs_dataset]
    title_array = ["High_School", "Primary_School", "US_Airport500", "Network_CoAuthors", "Open_Flights",
                   "DBLP", "Amazon", "Political_Blogs"]
    has_meta_array = [True, True, False, False, False, True, True, True]

    folderPath = 'Results/Datasets'
    with open(folderPath + '.txt', 'w', encoding="utf-8") as f:
        for i in range(len(get_dataset_array)):
            get_dataset = get_dataset_array[i]
            has_meta = has_meta_array[i]
            title = title_array[i]

            clusters_Meta = -1
            if has_meta:
                D, partition_Meta, B, T, S = get_graphs(get_dataset, has_meta, True)
                clusters_Meta = len(set(partition_Meta.values()))
            else:
                D, B, T, S = get_graphs(get_dataset, has_meta, True)

            graph_description_table = get_num_edges_table(D, B, S, clusters_Meta)
            f.write(title + '\n')
            f.write(graph_description_table)
            f.write('\n\n')


def compute_all_statistics(get_dataset, title, has_meta, from_contact, *args):
    partition_Meta = None
    clusters_Meta = -1
    if has_meta:
        partitionTypes = ['Metalabels', 'Louvain', 'Leiden', 'InfoMap', 'SVD_Laplacian_KMeans']
        D, partition_Meta, B, T, S = get_graphs(get_dataset, has_meta, from_contact, *args)
        clusters_Meta = len(set(partition_Meta.values()))
    else:
        partitionTypes = ['Louvain', 'Leiden', 'InfoMap', 'SVD_Laplacian_KMeans']
        D, B, T, S = get_graphs(get_dataset, has_meta, from_contact, *args)

    get_degree_distribution(D, title)
    get_degree_distribution(B, title + "_backbone")
    get_degree_distribution(T, title + "_threshold")
    check_backbone_removed_edge_distribution(D, B, T, title)

#    build_tables_clustering(D, B, T, title)
    graph_description_table = get_num_edges_table(D, B, T)

    partitions_D = get_all_partitions(D, clusters_Meta)
    partitions_B = get_all_partitions(B, clusters_Meta)
    partitions_T = get_all_partitions(T, clusters_Meta)

    get_incluster_degree_distribution(D, partitions_D['Leiden'][0], title)
    get_outcluster_degree_distribution(D, partitions_D['Leiden'][0], title)

    partition_table_D = get_original_partition_table(D, partitions_D, partitionTypes, partition_Meta, clusters_Meta)
    partition_table_B = get_comparison_partition_table(B, partitions_B, partitionTypes, partitions_D, partition_Meta, clusters_Meta)
    partition_table_T = get_comparison_partition_table(T, partitions_T, partitionTypes, partitions_D, partition_Meta, clusters_Meta)

    folderPath = 'Results/'
    with open(folderPath + title + '_partitions.txt', 'w', encoding="utf-8") as f:
        f.write(graph_description_table)
        f.write('\n\n')
        f.write("DISTANCE GRAPH (connected: " + str(nx.is_connected(D)) + ") \n")
        f.write(partition_table_D)
        f.write('\n\n')
        f.write("METRIC BACKBONE (connected: " + str(nx.is_connected(B)) + ") \n")
        f.write(partition_table_B)
        f.write('\n\n')
        f.write("THRESHOLD GRAPH (connected: " + str(nx.is_connected(T)) + ") \n")
        f.write(partition_table_T)


#compute_graphs_statistics()

#compute_all_statistics(get_high_school_dataset, "High School", True, True)
#compute_all_statistics(get_primary_school_dataset, "Primary_School", True, True)
#compute_all_statistics(get_DBLP_dataset, "DBLP", True, True)
