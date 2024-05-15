# thresholding-based subspace clustering (TSC) algorithm of Reinhard Heckel and Helmut Bölcskei

import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import SpectralClustering

from helper_plots import plot_TSC_k
from metrics import *
from graph_builder import *


def spectral_clustering(labels, X, numLabels):
    SC = SpectralClustering(numLabels, affinity="precomputed")
    pred_labels = SC.fit_predict(X)
    similarity = sklearn.metrics.adjusted_rand_score(labels, pred_labels)
    print("Similarity is %.2f%%" % (similarity * 100))
    return similarity


def TSC(A, k, labels, L):
    """
    Parameters
    ----------
    A: sparse weight matrix
    k: input parameter of TSC (for k-NN)
    labels: true labels of the nodes
    L: number of clusters, optional

    Returns
    -------
    similarity
    """
    start0 = time.time()
    numEdgesKNN = A.count_nonzero() // 2
    print("K-NN graph for k=" + str(k) + " has " + str(numEdgesKNN) + " edges")
    print("K-NN (k=" + str(k) + ") ", end="")

    startSC = time.time()
    sim = spectral_clustering(labels, A, L)
    end = time.time()

    print("Time to run Spectral Clustering: %.3f s" % (end - startSC), end="  ")
    print("TSC: %.3f s" % (end - start0))
    return sim, numEdgesKNN

def TSC_MB(A, k, labels, L, N, approx=False):
    """
    Parameters
    ----------
    A: sparse weight matrix
    k: input parameter of TSC (for k-NN)
    labels: true labels of the nodes
    L: number of clusters, optional
    N: number of samples
    Returns
    -------
    similarity
    """
    start0 = time.time()

    # Building metric backbone
    start = time.time()
    D = nx.Graph()
    D.add_nodes_from(range(N))
    rows, cols = A.nonzero()
    for u, v in zip(rows, cols):
        D.add_edge(u, v, proximity=A[u, v], weight=1 / A[u, v] - 1)

    if approx:
        B = get_approximate_metric_backbone_igraph(D)
    else:
        B = get_metric_backbone_igraph(D)

    A = nx.adjacency_matrix(B, nodelist=[i for i in range(N)], weight='proximity')
    A = scipy.sparse.csr_matrix(A)

    numEdgesMB = A.count_nonzero() // 2
    end = time.time()
    print("Time to build the backbone: %.3f s" % (end - start))
    print("K-NN's metric backbone has " + str(numEdgesMB) + " edges")
    print("Metric Backbone ", end="")

    startSC = time.time()
    sim = spectral_clustering(labels, A, L)
    end = time.time()

    print("Time to run Spectral Clustering: %.3f s" % (end - startSC), end="  ")
    print("TSC_MB: %.3f s" % (end - start0))
    return sim, numEdgesMB


# Spielman Method for Graph Sparsification
def TSC_SM(A, k, labels, L, numEdgesMB, approx=False, N=-1):
    """
    Parameters
    ----------
    A: sparse weight matrix
    k: input parameter of TSC (for k-NN)
    labels: true labels of the nodes
    L: number of clusters, optional
    numEdgesMB: number of edges in the MB and thus the spectral sparsifier

    Returns
    -------
    similarity
    """
    start0 = time.time()
    if not approx:
        GA = graphs.Graph(A)
        Gs = spectral_graph_sparsify(GA, numEdgesMB)

        numEdgesSM = Gs.W.count_nonzero() // 2
        print("Spielman Sparsification ", end="")

        startSC = time.time()
        sim = spectral_clustering(labels, Gs.W, L)
        end = time.time()
    else:
        D = nx.Graph()
        D.add_nodes_from(range(N))
        rows, cols = A.nonzero()
        for u, v in zip(rows, cols):
            D.add_edge(u, v, proximity=A[u, v], weight=1 / A[u, v] - 1)

        Gs = approximate_Spectral_Sparsifier(D)
        W_S = nx.adjacency_matrix(Gs, nodelist=[i for i in range(D.number_of_nodes())], weight='proximity')

        numEdgesSM = W_S.count_nonzero() // 2
        print("Spielman Sparsification ", end="")

        startSC = time.time()
        sim = spectral_clustering(labels, W_S, L)
        end = time.time()

    print("Time to run Spectral Clustering: %.3f s" % (end - startSC), end="  ")
    print("TSC_SM: %.3f s" % (end - start0))
    return sim, numEdgesSM


def compare_TSC_k(dataset, gaussianSimilarity=False, get_dataset=None, full_dataset=False):
    random.seed(11)

    numIter = 10
    numEdgesKNN = []
    numEdgesKNNStd = []
    numEdgesB = []
    numEdgesBStd = []
    numEdgesS = []
    numEdgesSStd = []
    allValsKNN = []
    allStdKNN = []
    allValsB = []
    allStdB = []
    allValsS = []
    allStdS = []
    ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for k in ks:
        curValsKNN = []
        curValsB = []
        curValsS = []
        curEdgesKNN = []
        curEdgesB = []
        curEdgesS = []
        for i in range(numIter):
            print('')
            if get_dataset is None:
                samples, labels, numLabels = get_gl_dataset(dataset, n=10000)
            else:
                samples, labels, numLabels = get_dataset()
            try:
                if not gaussianSimilarity:
                    A = get_cos_weight_matrix(samples.T, k)
                else:
                    A = get_Gaussian_weight_matrix(samples, k)

                if i > 0 and full_dataset:
                    valsS, edgesS = TSC_SM(A, k, labels, numLabels, curEdgesB[0])
                    curValsS.append(valsS)
                    curEdgesS.append(edgesS)
                    continue

                valsKNN, edgesKNN = TSC(A, k, labels, numLabels)
                if valsKNN == -1:
                    continue
                valsB, edgesB = TSC_MB(A, k, labels, numLabels, samples.shape[0])
                if valsB == -1:
                    continue
                valsS, edgesS = TSC_SM(A, k, labels, numLabels, edgesB)
                if valsS == -1:
                    continue
            except Exception as e:
                    print(e)
                    continue

            curValsB.append(valsB)
            curValsKNN.append(valsKNN)
            curValsS.append(valsS)
            curEdgesB.append(edgesB)
            curEdgesKNN.append(edgesKNN)
            curEdgesS.append(edgesS)

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
        numEdgesS.append(np.mean(curEdgesS))
        numEdgesSStd.append(np.std(curEdgesS))

    allVals = [allValsKNN, allStdKNN, allValsB, allStdB, allValsS, allStdS]
    numEdges = [numEdgesKNN, numEdgesKNNStd, numEdgesB, numEdgesBStd, numEdgesS, numEdgesSStd]
    path = "Results/Unsupervised/" + dataset + "_k_add.txt"
    if gaussianSimilarity:
        path = "Results/Unsupervised/" + dataset + "_k_GaussianSimilarity_add.txt"
    f = open(path, 'w', encoding="utf-8")
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


def test_large_n(dataset, gaussianSimilarity=False, file_MB="", isHAR=False):
    if isHAR:
        samples, labels, numLabels = get_HAR_dataset()
    else:
        samples, labels, numLabels = get_gl_dataset(dataset, fixedSeed=True, symmetricClusters=False)
    N = len(samples)
    k_sqrt = int(np.sqrt(N)/2)

    print("k_sqrt=" + str(k_sqrt))
    if not gaussianSimilarity:
        A10 = get_cos_weight_matrix(samples.T, 10)
    else:
        A10 = get_Gaussian_weight_matrix(samples, 10)
    valsKNN, edgesKNN = TSC(A10, 10, labels, numLabels)

    if file_MB == "":
        if not gaussianSimilarity:
            Asqrt = get_cos_weight_matrix(samples.T, k_sqrt)
        else:
            Asqrt = get_Gaussian_weight_matrix(samples, k_sqrt)
        valsB, edgesB = TSC_MB(Asqrt, k_sqrt, labels, numLabels, N, approx=True)
    else:
        B = readGraphFromFile(file_MB)

        A = nx.adjacency_matrix(B, nodelist=[i for i in range(N)], weight='proximity')
        A = scipy.sparse.csr_matrix(A)

        edgesB = A.count_nonzero() // 2
        print("K-NN's metric backbone has " + str(edgesB) + " edges")
        print("Metric Backbone ", end="")

        startSC = time.time()
        valsB = spectral_clustering(labels, A, numLabels)
        end = time.time()
        print("Time to run Spectral Clustering: %.3f s" % (end - startSC), end="  ")

    if not gaussianSimilarity:
        f = open("./" + dataset + "_k_TSC_comp.txt", 'w', encoding="utf-8")
    else:
        f = open("./" + dataset + "_k_TSC_Gaussian_comp.txt", 'w', encoding="utf-8")

    f.write(dataset + "\n")
    f.write(str(valsKNN) + "\n")
    f.write(str(valsB) + "\n")
    f.write(str(edgesKNN) + " " + str(edgesB) + '\n')


def reproduce_results():
    if not os.path.isdir("Results"):
        os.mkdir("Results")
    if not os.path.isdir("Results/Unsupervised"):
        os.mkdir("Results/Unsupervised")

    compare_TSC_k("MNIST")
    compare_TSC_k("FashionMNIST")
    compare_TSC_k("HAR", get_dataset=get_HAR_dataset, full_dataset=True)
    plot_TSC_k("MNIST_k", True)
    plot_TSC_k("FashionMNIST_k", True)
    plot_TSC_k("HAR_k", True)

    compare_TSC_k("MNIST", True)
    compare_TSC_k("FashionMNIST", True)
    compare_TSC_k("HAR", gaussianSimilarity=True, get_dataset=get_HAR_dataset, full_dataset=True)
    plot_TSC_k("MNIST_k_GaussianSimilarity", True)
    plot_TSC_k("FashionMNIST_k_GaussianSimilarity", True)
    plot_TSC_k("HAR_k_GaussianSimilarity", True)

    test_large_n("MNIST", True)
    test_large_n("FashionMNIST", True)
    test_large_n("HAR", True, isHAR=True)
    #test_large_n("MNIST", True, "MBApprox3_MNIST_132.txt")
    #test_large_n("FashionMNIST", True, "MBApprox_FashionMNIST_132.txt")
