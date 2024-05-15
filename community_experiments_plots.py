from metrics import *
from datasets import *
from clustering import *
from graph_builder import *

SEED = 11
colors = {"Original Graph": '#d62728', "Metric Backbone": '#1f77b4', "Threshold Subgraph": '#ff7f0e', "Spectral Sparsifier": '#2ca02c'}


def get_similarities(f, type, partitions, partitions_D, partition_Meta=None):
    similarity_D = {}
    similarity_metaLabels = {}
    for algo in partitions:
        (partition, cluster) = partitions[algo]
        (partitionD, clusterD) = partitions_D[algo]
        similarity_D[algo] = get_partitions_similarity_ARI(partitionD, partition)
        similarity_metaLabels[algo] = get_partitions_similarity_ARI(partition_Meta, partition)

    if partition_Meta is not None:
        for algo in similarity_metaLabels.keys():
            f.write(type + " Meta " + algo + " " + str(similarity_metaLabels[algo]) + "\n")
    for algo in similarity_D.keys():
        f.write(type + " Original " + algo + " " + str(similarity_D[algo]) + "\n")


def compute_similarities_real_datasets():
    np.random.seed(SEED)
    get_dataset_array = [get_high_school_dataset, get_primary_school_dataset, get_USairport500_dataset,
                         get_network_coauthor_dataset, get_OpenFlights_dataset, get_DBLP_dataset,
                         get_Amazon_dataset, get_Political_Blogs_dataset, get_wikiSchool_dataset, get_Cite_Seer_dataset, get_Cora_dataset]
    title_array = ["High_School", "Primary_School", "US_Airport500", "Network_CoAuthors", "Open_Flights",
                   "DBLP", "Amazon", "Political_Blogs", "WikiSchool", "Cite_Seer", "Cora"]
    has_meta_array = [True, True, False, False, False, True, True, True, True, True, True]

    get_dataset_array = [get_high_school_dataset, get_primary_school_dataset, get_DBLP_dataset, get_Amazon_dataset]
    title_array = ["High_School", "Primary_School", "DBLP", "Amazon"]
    has_meta_array = [True, True, True, True]

    for i in range(len(get_dataset_array)):
        get_dataset = get_dataset_array[i]
        has_meta = has_meta_array[i]
        title = title_array[i]
        print("\nNow working with " + title)
        f = open("Results/Communities Similarities/Similarities_" + title + ".txt", 'w', encoding="utf-8")

        partition_Meta = None
        clusters_Meta = -1
        if has_meta:
            D, partition_Meta, B, T, S = get_graphs(get_dataset, has_meta, True)
            clusters_Meta = len(set(partition_Meta.values()))
        else:
            D, B, T, S = get_graphs(get_dataset, has_meta, True)

        partitions_D = get_all_partitions(D, clusters_Meta)
        partitions_B = get_all_partitions(B, clusters_Meta)
        partitions_T = get_all_partitions(T, clusters_Meta)
        partitions_S = get_all_partitions(S, clusters_Meta)

        get_similarities(f, "Original", partitions_D, partitions_D, partition_Meta)
        get_similarities(f, "Backbone", partitions_B, partitions_D, partition_Meta)
        get_similarities(f, "Threshold", partitions_T, partitions_D, partition_Meta)
        get_similarities(f, "Spielman", partitions_S, partitions_D, partition_Meta)

        f.close()


def build_plots(vs_meta=False):
    algorithms = ["Leiden", "SVD_Laplacian_KMeans", "Bayesian"]
    names = ["Backbone", "Threshold", "Spielman"]
    if vs_meta:
        datasets_path = ["High_School", "Primary_School", "DBLP", "Amazon"]
        names.insert(0, "Original")
    else:
        datasets_path = ["US_Airport500", "DBLP", "Open_Flights", "Amazon"]

    for algo in algorithms:
        values = np.empty(shape=(len(names), len(datasets_path)))
        for j in range(len(datasets_path)):
            f = open("Results/Communities Similarities/Similarities_" + datasets_path[j] + ".txt", "r")
            lines = f.readlines()
            for line in lines:
                if algo not in line:
                    continue

                line = line.split()
                if vs_meta and line[1] != "Meta":
                    continue

                for i in range(len(names)):
                    if line[0] == names[i]:
                        val = float(line[-1])
                        values[i][j] = max(val, 0)
                        break
            f.close()
        plot_bars(values, algo, vs_meta)


def plot_bars(values, title, vs_meta=False):
    barWidth = 0.1
    if vs_meta:
        barWidth = 0.1

    bar_titles = ["Metric Backbone", "Threshold Subgraph", "Spectral Sparsifier"]
    if vs_meta:
        bar_titles.insert(0, "Original Graph")
        datasets_title = ["High School", "    Primary School", "DBLP", "Amazon"]
    else:
        datasets_title = ["US Airport 500", "DBLP", "Open Flights", "Amazon"]
    num_bars = len(bar_titles)

    br = [0] * num_bars
    br[0] = list(np.arange(len(datasets_title)))

    for i in range(1, num_bars):
        br[i] = [x + barWidth for x in br[i - 1]]

    for i in range(num_bars):
        curName = bar_titles[i]
        plt.bar(br[i], values[i], capsize=2.5, width=barWidth,
                edgecolor='grey', label=curName, color=colors[curName])

    vert_label = "ARI"
    if vs_meta:
        vert_label = "ARI"
 #   plt.title("Similarity Results for the " + title + " Algorithm")
    plt.xlabel("Dataset", fontsize=15)
    plt.ylabel(vert_label, fontsize=15)
    plt.rcParams['font.size'] = '15'
    plt.xticks([r + barWidth for r in range(len(datasets_title))], datasets_title)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tick_params(axis='both', which='major', labelsize=14.5)
    plt.tick_params(axis='both', which='minor', labelsize=14.5)
    plt.ylim(top=1.2)

    plt.legend()
    if vs_meta:
        fig_name = title + "_Meta.pdf"
    else:
        fig_name = title + "_Original.pdf"
    plt.savefig("Results/Communities Similarities/" + fig_name, format="pdf")
    plt.show()


def reproduce_results():
    if not os.path.isdir("Results"):
        os.mkdir("Results")
    if not os.path.isdir("Results/Communities Similarities"):
        os.mkdir("Results/Communities Similarities")

    compute_similarities_real_datasets()
    build_plots(True)

