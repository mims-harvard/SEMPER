from collections import Counter
from typing import Dict, List
import numpy as np
import pandas as pd
import json, matplotlib, os

import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

from extract_txdata_utils import *


def get_labels_from_evidence(disease: str, evidence_dir: str, all_drug_targets_path: str, curated_disease_dir: str, chembl2db_path: str, 
                             raw_data_prefix: str, overwrite: bool, disease_drug_evidence_prefix = "", wandb = None):
    """
    Get positive and negative genes/targets associated with each disease and descendants.
    """
    
    # Read in CHEMBL data
    chembl_db_df = pd.read_table(chembl2db_path) 
    chembl_db_df.columns = ['chembl', 'db']
    chembl2db = chembl_db_df.set_index('chembl').to_dict()['db']

    # Read in all approved drug-target data
    dti_tab = pd.read_csv(all_drug_targets_path, index_col=0)  # approved drug-target table
    assert dti_tab['Drug IDs'].isna().sum()==0
    dti_tab = dti_tab[dti_tab.Species=='Humans']
    druggable_targets = dti_tab[['Gene Name', 'GenAtlas ID']]
    druggable_targets = set(druggable_targets.values.flatten())
    druggable_targets.remove(np.nan)  # all approved drugs' targets

    evidence_files = os.listdir(evidence_dir)
    
    # Get all disease descendants (we include indirect evidence)
    all_disease = get_disease_descendants(disease, source='ot', curated_disease_dir=curated_disease_dir)
    if wandb is not None:
        wandb.log({f'number of disease descendants':len(all_disease)})
    
    # Look for clinically relevant evidence on targets related to each of the diseases.
    disease_drug_evidence_data = get_all_drug_evidence(evidence_files, evidence_dir, all_disease, chembl2db)
    print("Disease-drug evidence:", disease_drug_evidence_data.shape)
    print("Number of unique genes:", len(disease_drug_evidence_data["targetId"].unique()))
    print("Number of unique drugs:", len(disease_drug_evidence_data["drugId"].unique()))

    # Get all associated targets of disease
    all_associated_genes, ensg2otgenename = get_all_associated_genes(disease)
    all_associated_genes = set(all_associated_genes)
    print("Number of all associated targets:", len(all_associated_genes))

    # Convert clinically relevant targets to gene names
    disease_drug_targets = evidence2genename(disease_drug_evidence_data, ensg2otgenename)

    # Saving disease/drug-target evidence
    if disease_drug_evidence_prefix != "":

        # Clinically relevant targets
        disease_drug_evidence_data["targetId_genename"] = disease_drug_evidence_data["targetId"].map(ensg2otgenename)
        print(disease_drug_evidence_data)
        print(Counter("Phase " + str(a) + "," + str(b) for a, b in zip(disease_drug_evidence_data["clinicalPhase"].tolist(), disease_drug_evidence_data["clinicalStatus"].tolist())))
        print(disease_drug_evidence_data["drugId"].unique())
        disease_drug_evidence_data.to_csv(disease_drug_evidence_prefix + disease + ".csv", index = False, sep = "\t")

        # All associated targets
        all_targets = {
                        "all_associated_genes": list(all_associated_genes),
                        "all_druggable_targets": list(druggable_targets),
                        "clinically_relevant_targets": list(disease_drug_targets),
                        }
        for k, v in all_targets.items():
            print(k, len(v))
        with open(raw_data_prefix + disease + '.json', 'w') as f:
            json.dump(all_targets, f)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--celltype_ppi", type=str, default="../data/networks/ppi_edgelists/", help="Filename (prefix) of cell type PPI.")
    parser.add_argument('--disease', type=str)
    parser.add_argument('--evidence_dir', type=str) # Download OpenTargets ChEMBL evidence
    parser.add_argument('--all_drug_targets_path', type=str, default="../data/therapeutic_target_task/all_approved.csv") # Download DrugBank drug-target all approved
    parser.add_argument('--curated_disease_dir', type=str) # Download OpenTargets disease/phenotype evidence
    parser.add_argument('--chembl2db_path', type=str)  # Download mapping from ChEMBL id to DrugBank id from https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt
    parser.add_argument('--disease_drug_evidence_prefix', type=str, default="../data/therapeutic_target_task/disease_drug_evidence_")
    parser.add_argument('--raw_data_prefix', type=str, default="../data/therapeutic_target_task/raw_targets_")
    args = parser.parse_args()

    get_labels_from_evidence(args.disease, args.evidence_dir, args.all_drug_targets_path, args.curated_disease_dir, args.chembl2db_path, args.raw_data_prefix, overwrite = True, disease_drug_evidence_prefix = args.disease_drug_evidence_prefix)


if __name__ == '__main__':
    main()
