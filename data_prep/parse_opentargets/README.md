New packages

`conda install conda-forge::pyarrow`
`conda install conda-forge::fastparquet`
`conda install conda-forge::python-snappy`



Pyarrow does not fit with the current PINNACLE env. Creating a new one.

```
conda create -n opentargets_env pyarrow pandas snappy
conda install conda-forge::matplotlib
conda install conda-forge::seaborn
conda install anaconda::networkx
conda install conda-forge::pytorch
conda install anaconda::requests
conda install bioconda::mygene
```

`conda activate opentargets_env`


Directory of OpenTargets data (downloaded June 24, 2025)

`/n/netscratch/mzitnik_lab/Lab/mli/opentargets_24Jun2025/evidence`


`wget --recursive --no-parent --no-host-directories --cut-dirs 6 ftp://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/output/disease .`



DrugBank approved drug targets data (downloaded June 30, 2025)

`/n/netscratch/mzitnik_lab/Lab/mli/drugbank/all_approved.csv`

CHEMBL to DrugBank mapping:

`wget https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src2.txt.gz`


COMMAND to download neurodegenerative diseases data

```
python prepare_txdata.py \
    --disease MONDO_0004976 \
    --evidence_dir /n/netscratch/mzitnik_lab/Lab/mli/opentargets_24Jun2025/evidence/sourceId=chembl/ \
    --curated_disease_dir /n/netscratch/mzitnik_lab/Lab/mli/opentargets_24Jun2025/disease/ \
    --all_drug_targets_path /n/netscratch/mzitnik_lab/Lab/mli/drugbank/all_approved.csv \
    --chembl2db_path /n/netscratch/mzitnik_lab/Lab/mli/drugbank/chembl2drugbank.txt \
    --disease_drug_evidence_prefix /n/holylfs06/LABS/mzitnik_lab/Lab/mli/CNS_PINN/data/ \
    --raw_data_prefix /n/holylfs06/LABS/mzitnik_lab/Lab/mli/CNS_PINN/data/raw_targets_
```

Nervous system disease: EFO_0000618
    Table
        Number of unique drugs: 1309
        Number of unique genes: 841
    Json
        all_associated_genes 15670
        all_druggable_targets 2375
        clinically_relevant_targets 841

Multiple sclerosis: MONDO_0005301
    Table
        Number of unique genes: 168
        Number of unique drugs: 125
    Json
        all_associated_genes 1606
        all_druggable_targets 2375
        clinically_relevant_targets 168

Alzheimer's disease: MONDO_0004975
    Table
        Number of unique drugs: 127
        Number of unique genes: 133
    Json
        all_associated_genes 4584
        all_druggable_targets 2375
        clinically_relevant_targets 133

Parkinson disease: MONDO_0005180
    Table
        Number of unique drugs: 110
        Number of unique genes: 98
    Json
        all_associated_genes 1746
        all_druggable_targets 2375
        clinically_relevant_targets 98

Amyotrophic lateral sclerosis: MONDO_0004976
    Table
        Number of unique drugs: 51
        Number of unique genes: 99
    Json
        all_associated_genes 1089
        all_druggable_targets 2375
        clinically_relevant_targets 99

Huntington disease: MONDO_0007739
    Table
        Number of unique drugs: 19
        Number of unique genes: 86
    Json
        all_associated_genes 293
        all_druggable_targets 2375
        clinically_relevant_targets 86






Neurodegenerative disease: EFO_0005772
    Table
        Number of unique drugs: 323
        Number of unique genes: 377
    Json
        all_associated_genes 10735
        all_druggable_targets 2375
        clinically_relevant_targets 377




