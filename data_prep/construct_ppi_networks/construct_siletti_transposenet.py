import os
import glob
import argparse
import pandas as pd
import numpy as np
import json

from collections import Counter

import networkx as nx

from utils import load_global_PPI
from utils import calculate_correlation


def read_mappers(mapper):
    reverse_gene_mapper = json.load(open(mapper, "r"))
    if "pref" in mapper: reverse_gene_mapper = {k: [v] for k, v in reverse_gene_mapper.items()}
    print(mapper)

    # Sanity check
    for k, v in reverse_gene_mapper.items():
        if "pref" in mapper: assert len(v) == 1
        if len(v) > 1: print("Has more than one HGCN:", k, v)

    gene_mapper = dict()
    for k, v in reverse_gene_mapper.items():
        for v_i in v:
            if v_i in gene_mapper:
                gene_mapper[v_i].append(k)
            else:
                gene_mapper[v_i] = [k]
    
    # Sanity check
    for k, v in gene_mapper.items():
        if "pref" in mapper: assert len(v) == 1
        if len(v) > 1: print("Has more than one ENSG:", k, v)

    return reverse_gene_mapper, gene_mapper


def extract_celltype_ppi(input_f, output_f, ppi, mapper, degree_filter = -1, lcc = True):
    unmatched_genes = []
    reverse_gene_mapper, gene_mapper = read_mappers(mapper)

    # Read list of cell type specific genes (table)
    ranked_celltype_genes = pd.read_csv(input_f, sep = "\t")
    print(ranked_celltype_genes)
    celltype_ppi = dict.fromkeys([c.split("_")[0] for c in ranked_celltype_genes.columns])

    if output_f != "":
        out = open(output_f + ".csv", "w")

    min_degrees_per_network = []
    max_degrees_per_network = []
    median_degrees_per_network = []
    mean_degrees_per_network = []
    for i, c in enumerate(sorted(celltype_ppi)):
        c_df_sig_genes = ranked_celltype_genes[c].tolist()
        #c_df_sig_genes = c_df_sig_genes[:3000] # Set threshold again for the construction of PPI networks

        unmatched_genes.extend([g for g in c_df_sig_genes if g not in gene_mapper and g not in unmatched_genes])

        # Convert significant genes from HGNC to ENSEMBL
        new_c_df_sig_genes = []
        for g in c_df_sig_genes:
            if g in gene_mapper:
                new_c_df_sig_genes.extend(gene_mapper[g])
        print(len(c_df_sig_genes))
        print(len(new_c_df_sig_genes))
        
        # Extract subgraph
        c_subgraph = ppi.subgraph(new_c_df_sig_genes)
        print(c_subgraph)

        # Filter degrees
        if degree_filter != -1:
            c_subgraph_filtered = [g for g in c_subgraph.nodes() if c_subgraph.degree[g] <= degree_filter]
            c_subgraph = ppi.subgraph(c_subgraph_filtered)
            print(c_subgraph)
        
        if lcc:
            c_lcc = max(nx.connected_components(c_subgraph), key=len)
            print("Size of LCC for %s:" % c, len(c_lcc), "(%.2f of sig genes in PPI)" % (len(c_lcc) / len(c_subgraph)))
            if len(c_lcc) < 1000: continue

            c_graph = ppi.subgraph(c_lcc)
            assert nx.is_connected(c_graph)
            c_degrees = [c_graph.degree[c_node] for c_node in c_graph.nodes()]
            min_degrees_per_network.append(min(c_degrees))
            max_degrees_per_network.append(max(c_degrees))
            median_degrees_per_network.append(np.median(c_degrees))
            mean_degrees_per_network.append(np.mean(c_degrees))

        else:
            c_lcc = c_subgraph.nodes()
        
        # Write cell type specific PPI network genes
        sorted_c_lcc = sorted(list(c_lcc))
        celltype_ppi[c] = sorted_c_lcc
        nodes = ",".join(sorted_c_lcc)
        if output_f != "":
            out.write("\t".join([str(i), c, nodes]) + "\n")

    print("Unmatched genes")
    print(len(unmatched_genes))
    print(len(set(unmatched_genes)))

    print("Min deg per network:", np.mean(min_degrees_per_network), np.std(min_degrees_per_network))
    print("Max deg per network:", np.mean(max_degrees_per_network), np.std(max_degrees_per_network))
    print("Median deg per network:", np.mean(median_degrees_per_network), np.std(median_degrees_per_network))
    print("Mean deg per network:", np.mean(mean_degrees_per_network), np.std(mean_degrees_per_network))

    return celltype_ppi


def read_ppi(global_ppi_f, celltype_ppi_f):
    G = load_global_PPI(global_ppi_f)
    ppi_layers = dict()
    with open(celltype_ppi_f) as fin:
        for lin in fin:
            cluster = lin.split("\t")[1]
            ppi = lin.strip().split("\t")[2].split(",")
            ppi_layers[cluster] = G.subgraph(ppi)
            assert nx.is_connected(ppi_layers[cluster])
    return ppi_layers


def write_ppi_edgelists(ppi_layers, output_f):
    saved_edgelists = []
    for celltype, ppi in ppi_layers.items():
        output_edgelist_f = output_f + "_".join(celltype.split(" ")) + "_edgelist.txt"
        assert " " not in output_edgelist_f
        assert output_edgelist_f not in saved_edgelists
        print("Writing to...", output_edgelist_f)
        nx.write_edgelist(ppi, output_edgelist_f, data = False)
        saved_edgelists.append(output_edgelist_f)
    print("Finished writing %d edgelists." % len(saved_edgelists))

    # Create a single file containing the list of edgelists in the folder
    inventory_f = output_f + "inventory.txt"
    output_list_of_files = open(inventory_f, "w")
    for l in saved_edgelists:
        output_list_of_files.write(l + "\n")
    print("Saved inventory at %s" % inventory_f)


def trim_metagraph(ppi_layers, mg_prefix, mg_suffix):
    metagraph = nx.read_edgelist(mg_prefix + ".txt", data = False, delimiter = "\t", create_using = nx.DiGraph)
    keep_nodes = []
    for n in metagraph.nodes():
        if "cluster:" not in n: keep_nodes.append(n)
        else:
            if n.split(":")[1] in ppi_layers: keep_nodes.append(n)
            else: print("Missing %s" % n)
    new_metagraph = metagraph.subgraph(keep_nodes)
    print(new_metagraph)
    assert nx.is_connected(new_metagraph.to_undirected())

    nx.write_edgelist(new_metagraph, mg_prefix + "_" + mg_suffix, data = False, delimiter = "\t")


"""
# Neuron only (old)
python construct_siletti_transposenet.py -celltype_ppi_filename ../data/siletti_transposenet_degree500 -degree_filter 500 -celltype_ppi_dir ../data/networks/ppi_edgelists_degree500/
python construct_siletti_transposenet.py -celltype_ppi_filename ../data/siletti_transposenet_threshold3000_degree500 -degree_filter 500 -celltype_ppi_dir ../data/networks/ppi_edgelists_threshold3000_degree500/

# Neuron + non_neuron (old)
python construct_siletti_transposenet.py -rank_filename ../data/siletti_supercluster_cluster_all -celltype_ppi_filename ../data/siletti_transposenet_all_threshold3000_degree500 -degree_filter 500 -celltype_ppi_dir ../data/networks/ppi_edgelists_all_threshold3000_degree500/ -mg_suffix threshold3000_degree500.txt
python construct_siletti_transposenet.py -rank_filename ../data/siletti_supercluster_cluster_all -celltype_ppi_filename ../data/siletti_transposenet_all_threshold3000 -celltype_ppi_dir ../data/networks/ppi_edgelists_all_threshold3000/ -mg_suffix threshold3000.txt

# Neuron + non_neuron (latest)
python construct_siletti_transposenet.py -rank_filename ../data/siletti_supercluster_cluster_all -celltype_ppi_filename ../data/siletti_transposenet_all_threshold4000 -celltype_ppi_dir ../data/networks/ppi_edgelists_all_threshold4000/ -mg_suffix threshold4000.txt
"""

def main():

    parser = argparse.ArgumentParser(description="Constructing cell type specific PPI networks.")
    parser.add_argument("-transposenet_ppi_dir", type=str, default="../data/TransposeNet_v3/", help="Filename (prefix) of ranked genes.")
    parser.add_argument("-rank_filename", type=str, default="../data/siletti_supercluster_cluster", help="Filename (prefix) of ranked genes.")
    parser.add_argument("-celltype_ppi_filename", type=str, default="../data/siletti_transposenet", help="Filename (prefix) of cell type specific PPI.")
    parser.add_argument("-celltype_ppi_dir", type=str, default="../data/networks/ppi_edgelists/", help="Directory to save cell type specific PPI.")
    parser.add_argument("-degree_filter", type=int, default=-1, help="Filter for degree.")
    parser.add_argument("-mg_suffix", type=str, default=".txt", help="Suffix for the trimmed metagraph.")
    args = parser.parse_args()

    if not os.path.exists(args.celltype_ppi_filename + ".csv"):

        # Read global PPI
        print("Loading in global PPI network...")
        ppi = load_global_PPI(args.transposenet_ppi_dir + "human_background.csv", weighted = 0.5)
        print("Finished loading in `ppi`...")

        # Create cell type specific PPI networks
        print("Extracting cell type specific PPI...")
        extract_celltype_ppi(args.rank_filename + ".csv", args.celltype_ppi_filename, ppi, args.transposenet_ppi_dir + "ENSP_to_pref.json", args.degree_filter)

    # Read and write PPI edgelists
    print("Reading and writing PPI edgelists...")
    ppi_layers = read_ppi(args.transposenet_ppi_dir + "human_background.csv", args.celltype_ppi_filename + ".csv")
    
    ppi_size = []
    for c, ppi in ppi_layers.items():
        ppi_size.append(len(ppi.nodes()))
    print("Number of PPI networks:", len(ppi_layers))
    print("Min number of nodes per network:", min(ppi_size))
    print("Max number of nodes per network:", max(ppi_size))
    print("Median number of nodes per network:", np.median(ppi_size))
    print("Average number of nodes per network:", np.mean(ppi_size), np.std(ppi_size))
    print("Total number of nodes:", sum(ppi_size))
    
    write_ppi_edgelists(ppi_layers, args.celltype_ppi_dir)

    # Make sure the metagraph is aligned with the final networks
    trim_metagraph(ppi_layers, "../data/networks/mg_edgelist_brainregion_all", args.mg_suffix)

    print("All finished!")
    

if __name__ == "__main__":
    main() 
