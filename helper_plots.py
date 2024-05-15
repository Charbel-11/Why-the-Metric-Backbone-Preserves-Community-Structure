from community_experiments_plots import colors
from metrics import *
from graph_builder import *


def draw_MB_vs_Threshold(get_dataset, title, has_meta, from_contact, *args):
    from clustering import get_partitions_Leiden

    if has_meta:
        D, D_ig, partition_Meta, B, B_ig, T, T_ig, S, S_ig = get_graphs(get_dataset, has_meta, from_contact, *args)
    else:
        D, D_ig, B, B_ig, T, T_ig, S, S_ig = get_graphs(get_dataset, has_meta, from_contact, *args)

    node_colors = []
    if has_meta:
        for i in range(D.number_of_nodes()):
            node_colors.append(partition_Meta[i])
    else:
        partition_Leiden, clusters_Leiden = get_partitions_Leiden(D_ig, 'proximity')
        for i in range(D.number_of_nodes()):
            node_colors.append(partition_Leiden[i])

    print("D IS CONNECTED:", nx.is_connected(D))
    print("T IS CONNECTED:", nx.is_connected(T))
    print("S IS CONNECTED:", nx.is_connected(S))

    pos = nx.spring_layout(B)
    plt.figure(1)
    nx.draw_networkx(D, pos=pos, width=0.05, alpha=0.75, node_size=15, node_color=node_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs T/" + title + "_Original_Graph.pdf", format="pdf")
#    plt.show()

    edge_colors = []
    for (u, v) in B.edges():
        if T.has_edge(u, v):
            edge_colors.append("grey")
        else:
            edge_colors.append("red")

    plt.figure(2)
    nx.draw_networkx(B, pos=pos, width=0.2, alpha=0.75, node_size=15, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs T/" + title + "_Metric_Backbone.pdf", format="pdf")
  #  plt.show()

    edge_colors = []
    for (u, v) in T.edges():
        if B.has_edge(u, v):
            edge_colors.append("grey")
        else:
            edge_colors.append("blue")

    plt.figure(3)
    nx.draw_networkx(T, pos=pos, width=0.2, alpha=0.75, node_size=15, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs T/" + title + "_Threshold_Subgraph.pdf", format="pdf")
  #  plt.show()

    plt.figure(4)
    nx.draw_networkx(T, width=0.2, alpha=0.75, node_size=15, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs T/" + title + "_Threshold Own Position.pdf", format="pdf")


def draw_MB_vs_Spielman(get_dataset, title, has_meta, from_contact, *args):
    from clustering import get_partitions_Leiden

    if has_meta:
        D, D_ig, partition_Meta, B, B_ig, T, T_ig, S, S_ig = get_graphs(get_dataset, has_meta, from_contact, *args)
    else:
        D, D_ig, B, B_ig, T, T_ig, S, S_ig = get_graphs(get_dataset, has_meta, from_contact, *args)

    node_colors = []
    if has_meta:
        for i in range(D.number_of_nodes()):
            node_colors.append(partition_Meta[i])
    else:
        partition_Leiden, clusters_Leiden = get_partitions_Leiden(D_ig, 'proximity')
        for i in range(D.number_of_nodes()):
            node_colors.append(partition_Leiden[i])

    print("D IS CONNECTED:", nx.is_connected(D), "Number of edges:", D.number_of_edges())
    print("B IS CONNECTED:", nx.is_connected(B), "Number of edges:", B.number_of_edges())
    print("T IS CONNECTED:", nx.is_connected(T), "Number of edges:", T.number_of_edges())
    print("S IS CONNECTED:", nx.is_connected(S), "Number of edges:", S.number_of_edges())

    pos = nx.spring_layout(B)
    plt.figure(1)
    nx.draw_networkx(D, pos=pos, width=0.05, alpha=0.75, node_size=15, node_color=node_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs Sp/" + title + "_Original_Graph.pdf", format="pdf")
#    plt.show()

    edge_colors = []
    for (u, v) in B.edges():
        if S.has_edge(u, v):
            edge_colors.append("grey")
        else:
            edge_colors.append("red")

    plt.figure(2)
    nx.draw_networkx(B, pos=pos, width=0.2, alpha=0.75, node_size=15, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs Sp/" + title + "_Metric_Backbone.pdf", format="pdf")
  #  plt.show()

    edge_colors = []
    for (u, v) in S.edges():
        if B.has_edge(u, v):
            edge_colors.append("grey")
        else:
            edge_colors.append("blue")

    plt.figure(3)
    nx.draw_networkx(S, pos=pos, width=0.2, alpha=0.75, node_size=15, node_color=node_colors, edge_color=edge_colors, with_labels=False)
    plt.savefig("Results/Graph Visualization MB vs Sp/" + title + "_Spectral_Sparsifier.pdf", format="pdf")
  #  plt.show()


def plot_curve(vals, inputPath, outputPath, isN=True, standardError=False, numTrials=-1):
    f = open(inputPath, 'r', encoding="utf-8")
    lines = f.readlines()

    div = 1
    if standardError:
        div = np.sqrt(numTrials)
    allVals = [[] for _ in range(6)]
    numEdges = [[] for _ in range(6)]
    for i in range(6):
        allVals[i] = lines[i].split(' ')
        if allVals[i][-1] == '\n':
            allVals[i].pop()
        allVals[i] = [float(x) for x in allVals[i]]
        if i % 2 == 1:
            allVals[i] /= div
    for i in range(6, 12):
        numEdges[i - 6] = lines[i].split(' ')
        if numEdges[i - 6][-1] == '\n':
            numEdges[i - 6].pop()
        numEdges[i - 6] = [float(x) for x in numEdges[i - 6]]

    plt.figure(1)
    plt.errorbar(vals, allVals[0], yerr=allVals[1], label="Original Graph", marker='.',
                 color=colors["Original Graph"], linewidth=3, elinewidth=1)
    plt.errorbar(vals, allVals[2], yerr=allVals[3], label="Metric Backbone", marker='.',
                 color=colors["Metric Backbone"], linewidth=3, elinewidth=1)
    plt.errorbar(vals, allVals[4], yerr=allVals[5], label="Spectral Sparsifier", marker='.',
                 color=colors["Spectral Sparsifier"], linewidth=3, elinewidth=1)
    if isN:
        plt.xlabel("n", fontsize=17)
    else:
        plt.xlabel("q", fontsize=17)
    plt.ylabel("ARI", fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.rcParams['font.size'] = '17'
    plt.legend()

    bottom, top = plt.ylim()
    if top - bottom < 0.1:
        diffNeeded = 0.1-(top - bottom)
        bottom -= diffNeeded/2
        top += diffNeeded/2
    plt.ylim((bottom, top))

    if isN:
        plt.savefig(outputPath + "_Similarity.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.savefig(outputPath + "_Similarity_k.pdf", format="pdf", bbox_inches="tight")

    plt.figure(2)
    plt.errorbar(vals, numEdges[0], yerr=numEdges[1], label="Original Graph", marker='.',
                 color=colors["Original Graph"])
    plt.errorbar(vals, numEdges[2], yerr=numEdges[3], label="Metric Backbone", marker='.',
                 color=colors["Metric Backbone"])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    if isN:
        plt.xlabel("n", fontsize=17)
    else:
        plt.xlabel("q", fontsize=17)
    plt.ylabel("Number of Edges", fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.rcParams['font.size'] = '17'
    plt.legend()
    if isN:
        plt.savefig(outputPath + "_Edges.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.savefig(outputPath + "_Edges_k.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def plot_TSC_k(dataset, standardError):
    ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plot_curve(ks, "Results/Unsupervised/" + dataset + ".txt", "Results/Unsupervised/" + dataset + "_TSC", isN=False, standardError=standardError, numTrials=10)


def plot_SSL_k(dataset, standardError):
    ks = [10, 20, 30, 40, 50, 60, 70, 80]
    plot_curve(ks, "Results/SSL/" + dataset + ".txt", "Results/SSL/" + dataset + "_SSL", isN=False, standardError=standardError, numTrials=100)


def mean_std(vals):
    print(str(np.mean(vals)) + " " + str(np.std(vals)))


#draw_MB_vs_Spielman(get_Political_Blogs_dataset, "Political Blogs", True, True)
#draw_MB_vs_Threshold(get_Political_Blogs_dataset, "Political Blogs", True, True)

