import sklearn
import time

from pygsp import graphs
from matplotlib import pyplot as plt

from datasets import *
from graph_builder import spectral_graph_sparsify, get_threshold_graph, approximate_Spectral_Sparsifier, \
    readGraphFromFile
from helper_plots import plot_SSL_k
from metric_backbone import get_metric_backbone_igraph, get_metric_backbone_igraph_slow, \
    get_approximate_metric_backbone_igraph

def runSSL(labels, W):
    vals = {}
    numIter = 10
    num_train_per_class = 10  # Number of seeds
    for i in range(numIter):
        train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
        train_labels = labels[train_ind]

        models = [gl.ssl.poisson(W)]
        for model in models:
            pred_labels = model.fit_predict(train_ind, train_labels)
            accuracy = sklearn.metrics.adjusted_rand_score(labels, pred_labels)
            if model.name not in vals:
                vals[model.name] = []
            vals[model.name].append(accuracy)

    for name in vals:
        print(name, "%.3f" % np.mean(vals[name]), "%.3f" % np.std(vals[name]), end='  ')
    return np.mean(vals['Poisson Learning']), np.std(vals['Poisson Learning'])


def compare_SSL(k, samples, labels):
    W0 = gl.weightmatrix.knn(samples, k)  # Gaussian similarity measure
    W0 = (W0 + W0.transpose()) / 2
    N = len(samples)

    print("K-NN (k=" + str(k) + ") Results:\t\t", end='')
    start = time.time()
    valsKNN = runSSL(labels, W0)
    end = time.time()
    timeOnKNN = end - start
    print("")

    # Building metric backbone
    start = time.time()
    D = nx.Graph()
    D.add_nodes_from(range(N))
    rows, cols = W0.nonzero()
    for u, v in zip(rows, cols):
        D.add_edge(u, v, proximity=W0[u, v], weight=1/W0[u, v]-1)

    B = get_metric_backbone_igraph(D)
    W = nx.adjacency_matrix(B, nodelist=[i for i in range(N)], weight='proximity')
    end = time.time()
    backboneBuildTime = end - start

    print("Metric Backbone Results:\t", end='')
    start = time.time()
    valsB = runSSL(labels, W)
    end = time.time()
    timeOnB = end - start
    print("")

    start = time.time()
    GA = graphs.Graph(W0)
    Gs = spectral_graph_sparsify(GA, B.number_of_edges())
    end = time.time()
    SpBuildTime = end-start

    print("Spielman Sparsify Results:\t", end='')
    start = time.time()
    valsS = runSSL(labels, Gs.W)
    end = time.time()
    timeOnSp = end-start
    print("")

    print("Time to build the backbone: %.2f s" % backboneBuildTime, end="; ")
    print("Time to build Spielman Sparsifier: %2.f s" % SpBuildTime)

    print("Time to run SSL (s):  K-NN %.2f,  MB %.2f,  Sp %.2f" % (timeOnKNN, timeOnB, timeOnSp))

    numEdgesKNN = W0.count_nonzero() // 2
    numEdgesSM = Gs.W.count_nonzero() // 2
    print("Number of Edges:  K-NN %s,  MB %s,  Sp %s" % (numEdgesKNN, B.number_of_edges(), numEdgesSM))

    print("")
    return valsKNN, valsB, valsS, numEdgesKNN, B.number_of_edges(), numEdgesSM


def test_semisupervised(dataset, get_dataset=None):
    numEdgesKNN = []
    numEdgesKNNStd = []
    numEdgesB = []
    numEdgesBStd = []
    allValsKNN = []
    allStdKNN = []
    allValsB = []
    allStdB = []
    allValsS = []
    allStdS = []

    numIter = 10
    ks = [10, 20, 30, 40, 50, 60, 70, 80]
    for k in ks:
        curValsKNN = []
        curValsB = []
        curValsS = []
        curEdgesKNN = []
        curEdgesB = []
        for i in range(numIter):
            if get_dataset == None:
                samples, labels, numLabels = get_gl_dataset(dataset, n=10000)
            else:
                samples, labels, numLabels = get_dataset()
            valsKNN, valsB, valsS, edgesKNN, edgesB, edgesS = compare_SSL(k, samples, labels)
            if valsS[0] == -1:
                continue

            curValsB.append(valsB[0])
            curValsKNN.append(valsKNN[0])
            curValsS.append(valsS[0])
            curEdgesB.append(edgesB)
            curEdgesKNN.append(edgesKNN)

        allValsKNN.append(np.mean(curValsKNN))
        allStdKNN.append(np.std(curValsKNN))
        allValsB.append(np.mean(curValsB))
        allStdB.append(np.std(curValsB))
        allValsS.append(np.mean(curValsS))
        allStdS.append(np.std(curValsS))
        numEdgesKNN.append(np.mean(curEdgesKNN))
        numEdgesKNNStd.append(np.std(curEdgesKNN))
        numEdgesB.append(np.mean(curEdgesB))
        numEdgesBStd.append(np.std(curEdgesB))

    allVals = [allValsKNN, allStdKNN, allValsB, allStdB, allValsS, allStdS]
    numEdges = [numEdgesKNN, numEdgesKNNStd, numEdgesB, numEdgesBStd, numEdgesB, numEdgesBStd]
    f = open("Results/SSL/" + dataset + "_k.txt", 'w', encoding="utf-8")
    for i in range(len(allVals)):
        for x in allVals[i]:
            f.write(str(x))
            f.write(' ')
        f.write('\n')

    for i in range(len(numEdges)):
        for x in numEdges[i]:
            f.write(str(x))
            f.write(' ')
        f.write('\n')

    plt.figure(1)
    plt.errorbar(ks, allValsKNN, yerr=allStdKNN, label="Original Graph", marker='.')
    plt.errorbar(ks, allValsB, yerr=allStdB, label="Metric Backbone", marker='.')
    plt.errorbar(ks, allValsS, yerr=allStdS, label="Spectral Sparsifier", marker='.')
    plt.xlabel("q")
    plt.ylabel("ARI")
    plt.legend()

    plt.figure(2)
    plt.errorbar(ks, numEdgesKNN, yerr=numEdgesKNNStd, label="Original Graph", marker='.')
    plt.errorbar(ks, numEdgesB, yerr=numEdgesBStd, label="Metric Backbone", marker='.')
    plt.errorbar(ks, numEdgesB, yerr=numEdgesBStd, label="Spectral Sparsifier", marker='.')
    plt.xlabel("q")
    plt.ylabel("Number of Edges")
    plt.legend()
    plt.show()


def test_large_n(dataset, file_MB="", isHAR=False):
    if isHAR:
        samples, labels, numLabels = get_HAR_dataset()
    else:
        samples, labels, numLabels = get_gl_dataset(dataset, fixedSeed=True, symmetricClusters=False)
    N = len(samples)

    W10 = gl.weightmatrix.knn(samples, 10)  # Gaussian similarity measure
    W10 = (W10 + W10.transpose()) / 2

    print("K-NN (k=10) Results:\t\t\t", end='')
    start = time.time()
    valsKNN = runSSL(labels, W10)
    end = time.time()
    timeOnKNN = end - start
    print("")

    k_sqrt = int(np.sqrt(N)/2)
    start = time.time()
    if file_MB == "":
        Wsqrt = gl.weightmatrix.knn(samples, k_sqrt)  # Gaussian similarity measure
        Wsqrt = (Wsqrt + Wsqrt.transpose()) / 2

        # Building metric backbone
        D = nx.Graph()
        D.add_nodes_from(range(N))
        rows, cols = Wsqrt.nonzero()
        for u, v in zip(rows, cols):
            D.add_edge(u, v, proximity=Wsqrt[u, v], weight=1 / Wsqrt[u, v] - 1)

        B = get_approximate_metric_backbone_igraph(D)
    else:
        # The file was pre-computed using C++ code
        B = readGraphFromFile(file_MB)

    end = time.time()
    backboneBuildTime = end - start
    W = nx.adjacency_matrix(B, nodelist=[i for i in range(N)], weight='proximity')

    print("Metric Backbone (k=" + str(k_sqrt) + ") Results:\t", end='')
    start = time.time()
    valsB = runSSL(labels, W)
    end = time.time()
    timeOnB = end - start
    print("")

    print("Time to build the backbone: %.2f s" % backboneBuildTime, end="; ")
    print("Time to run SSL (s):  K-NN %.2f,  MB %.2f" % (timeOnKNN, timeOnB))

    numEdgesKNN = W10.count_nonzero() // 2
    numEdgesB = B.number_of_edges()
    print("Number of Edges:  K-NN %s,  MB %s" % (numEdgesKNN, B.number_of_edges()))

    f = open("Results/SSL/" + dataset + "_k_comp.txt", 'w', encoding="utf-8")
    f.write(dataset + "\n")
    f.write(str(valsKNN[0]) + " " + str(valsKNN[1]) + "\n")
    f.write(str(valsB[0]) + " " + str(valsB[1]) + "\n")
    f.write(str(numEdgesKNN) + " " + str(numEdgesB))


def reproduce_results():
    if not os.path.isdir("Results"):
        os.mkdir("Results")
    if not os.path.isdir("Results/SSL"):
        os.mkdir("Results/SSL")

    test_semisupervised("MNIST")
    test_semisupervised("FashionMNIST")
    test_semisupervised("HAR", get_HAR_dataset)

    plot_SSL_k("MNIST_k", True)
    plot_SSL_k("FashionMNIST_k", True)
    plot_SSL_k("HAR_k", True)

    test_large_n("MNIST")
    test_large_n("FashionMNIST")
    test_large_n("HAR", isHAR=True)
    #test_large_n("MNIST", "MBApprox3_MNIST_132.txt")
    #test_large_n("FashionMNIST", "MBApprox_FashionMNIST_132.txt")
