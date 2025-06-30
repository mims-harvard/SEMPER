# Single cell transcriptomic atlas = Siletti et al

import os
import pandas as pd
import numpy as np
import networkx as nx


SILETTI_DIR = "/n/data1/hms/dbmi/zitnik/lab/datasets/2024-05-Siletti/processed/"


def check_no_overlap_keys(d1, d2):
    assert len(set(list(d1.keys())).intersection(set(list(d2.keys())))) == 0


def read_cluster_markers(f):
    supercluster_df = pd.read_csv(f)
    cluster_names = set([c.split("_")[0] for c in supercluster_df.columns])
    return supercluster_df, cluster_names


def map_superclusters2clusters(supercluster_names, folder_type):
    supercluster2cluster_mapping = dict()
    cluster2supercluster_mapping = dict()
    supercluster2df = dict()
    for n in supercluster_names:
        supercluster_n_f = SILETTI_DIR + folder_type + n + "_markers.csv"
        if not os.path.exists(supercluster_n_f):
            print("Does not exist:", supercluster_n_f)
            continue
        #supercluster_n_f = SILETTI_DIR + "neuron_markers/" + n + "_markers.csv"
        supercluster_n_df, cluster_n_names = read_cluster_markers(supercluster_n_f)
        
        assert n not in supercluster2cluster_mapping
        supercluster2df[n] = supercluster_n_df
        supercluster2cluster_mapping[n] = cluster_n_names
        
        for c in cluster_n_names:
            assert c not in cluster2supercluster_mapping
            cluster2supercluster_mapping[c] = n
    
    print(len(supercluster2cluster_mapping))
    print(len(cluster2supercluster_mapping))
    print(supercluster2cluster_mapping.keys())
    print(cluster2supercluster_mapping.keys())
    
    return supercluster2cluster_mapping, cluster2supercluster_mapping, supercluster2df


def map_brainregions(region2structure_f, stucture2cluster_f, cluster2supercluster_mapping):
    metagraph_edges = []

    # Get region-structure edges
    region_structure_df = pd.read_csv(region2structure_f)
    print(region_structure_df)

    for region in region_structure_df["x_id"].unique():
        metagraph_edges.append(("BTO:WholeBrain", "BTO:" + str(region)))
        metagraph_edges.append(("BTO:" + str(region), "BTO:WholeBrain"))
    print("Number of wholebrain-region edges:", len(metagraph_edges))

    for x, y in zip(region_structure_df["x_id"].tolist(), region_structure_df["y_id"].tolist()):
        metagraph_edges.append(("BTO:" + str(x), "BTO:" + str(y)))
        metagraph_edges.append(("BTO:" + str(y), "BTO:" + str(x)))
    
    # Get structure-cluster edges
    structure_cluster_df = pd.read_csv(stucture2cluster_f)
    print(structure_cluster_df)

    # Get structure-supercluster edges
    structure_cluster_df["supercluster"] = structure_cluster_df["y_id"].astype(str).map(cluster2supercluster_mapping)
    print(structure_cluster_df)
    print(structure_cluster_df[structure_cluster_df["supercluster"].isna()])
    structure_cluster_df = structure_cluster_df.dropna(subset=["supercluster"])
    print(structure_cluster_df)
    print(structure_cluster_df[["x_id", "supercluster"]].drop_duplicates())
    for x, y in zip(structure_cluster_df["x_id"].tolist(), structure_cluster_df["supercluster"].tolist()):
        metagraph_edges.append(("BTO:" + str(x), "BTO:" + str(y)))
        metagraph_edges.append(("BTO:" + str(y), "BTO:" + str(x)))

    metagraph = nx.DiGraph()
    metagraph.add_edges_from(metagraph_edges)
    print("Number of brain region related nodes:", len(metagraph.nodes()))
    print("Number of brain region related edges:", len(metagraph.edges()))
    return metagraph_edges


def get_topk_ranked_genes(metagraph_edges, supercluster2cluster_mapping, cluster2supercluster_mapping, superclusters_df, supercluster2df, max_pval = 0.01, max_number_of_genes = 4000): # max_number_of_genes = 3000
    supercluster_names = set([c.split("_")[0] for c in superclusters_df.columns])
    
    all_cluster_genes = dict()
    cluster_lengths = []
    for supercluster in supercluster_names:
        
        # Extract columns corresponding to supercluster
        sc_df = superclusters_df[[c for c in superclusters_df.columns if supercluster in c]]
        # Filter for significant genes in supercluster
        sc_df_sig = sc_df[sc_df[supercluster + "_pvals_adj"] <= max_pval]
        # Extract top genes from supercluster
        superclusters_genes = sc_df_sig[supercluster + "_names"].tolist()[:max_number_of_genes//2]
        
        # Get clusters associated with the supercluster
        if supercluster not in supercluster2cluster_mapping:
            print("Not found:", supercluster)
            continue
        
        clusters_names = supercluster2cluster_mapping[supercluster]
        # Add cluster-supercluster edges to the metagraph
        #metagraph_edges.extend([("BTO:" + supercluster, "BTO:neuron")] + [("BTO:" + supercluster, "cluster:" + c) for c in clusters_names]) # For no brain region
        metagraph_edges.extend([("BTO:" + supercluster, "cluster:" + c) for c in clusters_names])
        metagraph_edges.extend([("cluster:" + c, "BTO:" + supercluster) for c in clusters_names])
        
        # Extract data for clusters of supercluster
        clusters_df = supercluster2df[supercluster]
        # For each cluster
        for cluster in clusters_names:
            # Extract columns corresponding to cluster
            c_df = clusters_df[[c for c in clusters_df.columns if cluster in c]]
            # Filter for significant genes in cluster
            c_df_sig = c_df[c_df[cluster + "_pvals_adj"] <= max_pval]
            # Extract top genes from cluster
            c_genes = c_df_sig[cluster + "_names"].tolist()#[:max_number_of_genes]
            
            # Concatenate genes from supercluster and cluster
            c_concat_genes = superclusters_genes + [g for g in c_genes if g not in superclusters_genes] # Maintain order of the genes
            #assert len(c_concat_genes) >= max_number_of_genes, len(c_concat_genes)
            if len(c_concat_genes) < max_number_of_genes: continue
            # Apply cutoff for number of genes
            c_concat_genes = c_concat_genes[:max_number_of_genes]
            
            # Save cluster genes
            cluster_lengths.append(len(c_concat_genes))
            assert cluster not in all_cluster_genes, cluster
            all_cluster_genes[cluster] = c_concat_genes

    print("Number of clusters:", len(all_cluster_genes))
    print("Average size of cluster layers:", np.mean(cluster_lengths), np.std(cluster_lengths))
    
    metagraph = nx.DiGraph()
    metagraph.add_edges_from(metagraph_edges)
    print("Number of nodes:", len(metagraph.nodes()))
    print("Number of edges:", len(metagraph.edges()))
    undirected_metagraph = nx.Graph()
    undirected_metagraph.add_edges_from(metagraph_edges)
    assert nx.is_connected(undirected_metagraph)
    degrees = []
    for n in undirected_metagraph.nodes():
        n_deg = undirected_metagraph.degree[n]
        if n_deg > 1: print(n, n_deg)
        degrees.append(n_deg)
    print("Min degree:", min(degrees))
    print("Max degree:", max(degrees))
    print("Median degree:", np.median(degrees))
    print("Average degree:", np.mean(degrees), np.std(degrees))

    return metagraph, all_cluster_genes


def main():
    
    # Siletti superclusters
    #   _names = gene names
    #   _scores = z-score produced by scanpy.tl.rank_genes_groups
    #   _pvals = p-values
    #   _pvals_adj = corrected p-values
    #   logfoldchanges = logFC
    neuron_superclusters = SILETTI_DIR + "neuron_markers/supercluster_markers.csv"
    neuron_superclusters_df, neuron_supercluster_names = read_cluster_markers(neuron_superclusters)
    print(neuron_superclusters_df)
    print(neuron_supercluster_names)
    print("Number of neuron superclusters:", len(neuron_supercluster_names))
    non_neuron_superclusters = SILETTI_DIR + "non_neuron_markers/supercluster_markers.csv"
    non_neuron_superclusters_df, non_neuron_supercluster_names = read_cluster_markers(non_neuron_superclusters)
    print(non_neuron_superclusters_df)
    print(non_neuron_supercluster_names)
    print("Number of non-neuron superclusters:", len(non_neuron_supercluster_names))
    superclusters_df = pd.concat([neuron_superclusters_df, non_neuron_superclusters_df])
    print(superclusters_df)
    neuron_map = {s: "neuron" for s in neuron_supercluster_names}
    #non_neuron_map = {s: "non_neuron" for s in non_neuron_supercluster_names}
    #is_neuron_map = {**neuron_map, **non_neuron_map}
    #print(is_neuron_map)

    # Superclusters to clusters
    neuron_supercluster2cluster_mapping, neuron_cluster2supercluster_mapping, neuron_supercluster2df = map_superclusters2clusters(neuron_supercluster_names, "neuron_markers/")
    non_neuron_supercluster2cluster_mapping, non_neuron_cluster2supercluster_mapping, non_neuron_supercluster2df = map_superclusters2clusters(non_neuron_supercluster_names, "non_neuron_markers/")
    
    #metagraph = [] # For no brain region
    check_no_overlap_keys(neuron_supercluster2cluster_mapping, non_neuron_supercluster2cluster_mapping)
    check_no_overlap_keys(neuron_cluster2supercluster_mapping, non_neuron_cluster2supercluster_mapping)
    check_no_overlap_keys(neuron_supercluster2df, non_neuron_supercluster2df)
    print("Neurons:", len(neuron_supercluster2cluster_mapping), len(neuron_cluster2supercluster_mapping), len(neuron_supercluster2df))
    print("Non-neurons:", len(non_neuron_supercluster2cluster_mapping), len(non_neuron_cluster2supercluster_mapping), len(non_neuron_supercluster2df))
    supercluster2cluster_mapping = {**neuron_supercluster2cluster_mapping, **non_neuron_supercluster2cluster_mapping}
    cluster2supercluster_mapping = {**neuron_cluster2supercluster_mapping, **non_neuron_cluster2supercluster_mapping}
    supercluster2df = {**neuron_supercluster2df, **non_neuron_supercluster2df}
    print("Total:", len(supercluster2cluster_mapping), len(cluster2supercluster_mapping), len(supercluster2df))
    metagraph = map_brainregions("../data/brain_region/region_to_structure_edges.csv", "../data/brain_region/structure_to_cell_cluster_edges.csv", cluster2supercluster_mapping)

    # Get ranked markers per cell type
    #   - Top K from superclusters_df
    #   - Top J from supercluster2df.values()
    metagraph, all_cluster_genes = get_topk_ranked_genes(metagraph, supercluster2cluster_mapping, cluster2supercluster_mapping, superclusters_df, supercluster2df)

    # Write data
    #nx.write_edgelist(metagraph, "../data/networks/mg_edgelist.txt", data = False, delimiter = "\t") # For no brain region
    nx.write_edgelist(metagraph, "../data/networks/mg_edgelist_brainregion_all.txt", data = False, delimiter = "\t")
    
    celltype_dfs = pd.DataFrame.from_dict(all_cluster_genes)
    celltype_dfs.to_csv("../data/siletti_supercluster_cluster_all.csv", sep = "\t", index = False)
    




if __name__ == "__main__":
    main()
