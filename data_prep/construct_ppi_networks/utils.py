from collections import Counter
import pandas as pd
import numpy as np
#from sklearn.metrics import r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import obonet
import glob


def load_global_PPI(f, weighted = 0):
    # Read table from csv file
    graph_df = pd.read_csv(f)

    # Create a list of tuples, where each tuple is an edge
    if weighted > 0: # Apply threshold
        edges = [(s, t) for s, t, w in zip(graph_df["protein1"].tolist(), graph_df["protein2"].tolist(), graph_df["weight"].tolist()) if w >= weighted]
    else:
        edges = [(s, t) for s, t in zip(graph_df["protein1"].tolist(), graph_df["protein2"].tolist())]

    # Instantiate graph object
    G = nx.Graph()

    # Add edges (from the table) to the graph object
    G.add_edges_from(edges)

    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))

    return G


def load_celltype_ppi(ppi_dir):
    ppi_layers = dict()

    for f in glob.glob(ppi_dir + "*"): # Expected format of filename: <PPI_DIR>/<CONTEXT>.<suffix>
        if "inventory.txt" in f: continue

        # Parse name of context
        context = f.split(ppi_dir)[1].split(".")[0]
        context = context.replace("_", " ")

        # Read edgelist
        ppi = nx.read_edgelist(f)
        ppi_layers[context] = ppi
        assert nx.is_connected(ppi_layers[context])

    print(ppi_layers.keys())
    return ppi_layers


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def ontology_distance(tree, a, b, diameter):
    if not nx.has_path(tree, a, b): return None
    return nx.shortest_path_length(tree, a, b) / diameter


def read_obo(f):
    return obonet.read_obo(f)


def calculate_correlation(x, y, title, x_label, y_label, plot_f):
    x_ordered = []
    y_ordered = []
    for pair in x:
        x_ordered.append(x[pair])
        y_ordered.append(y[pair])

    ax = sns.regplot(x = x_ordered, y = y_ordered, marker = ".", y_jitter = 0.005, scatter_kws={"color": "silver"}, line_kws={"color": "darkcyan"})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_f)
    plt.close()
    return spearmanr(x_ordered, y_ordered)


def plot_box(data, dodge, x, y, title, x_label, y_label, legend, plot_f):
    ax = sns.boxplot(data = data, x = x, y = y, hue = x, dodge = dodge)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    if not legend: ax.get_legend().remove()
    plt.savefig(plot_f)
    plt.close()
