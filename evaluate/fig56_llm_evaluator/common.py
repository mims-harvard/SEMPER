"""Shared helpers for the Figure 5-6 LLM evaluator scripts.

Credentials are read from environment variables only. Nothing in this folder
stores an API key.

  Azure OpenAI (what the paper used):
      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
  OpenAI (fallback, used only if the Azure variables are not set):
      OPENAI_API_KEY
"""

import csv
import os
import sys

# Deployment used for every LLM call in Fig 5d-j and Fig 6.
PAPER_MODEL = "gpt-5.6-sol"

# abbreviation -> (output tag, disease name used inside the judge / referee prompts)
DISEASES = {
    "pd":  ("PD",  "Parkinson's Disease"),
    "ad":  ("AD",  "Alzheimer's Disease"),
    "ms":  ("MS",  "Multiple Sclerosis (MS)"),
    "als": ("ALS", "Amyotrophic Lateral Sclerosis (ALS)"),
    "hd":  ("HD",  "Huntington's Disease"),
}


def resolve_disease(value):
    """'pd' -> "Parkinson's Disease"; any other string is used verbatim."""
    if value.lower() in DISEASES:
        return DISEASES[value.lower()][1]
    return value


def make_client(api_version):
    """Return a chat-completions client built from environment variables."""
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        from openai import AzureOpenAI
        return AzureOpenAI(azure_endpoint=azure_endpoint, api_key=azure_key,
                           api_version=api_version)
    if azure_key or azure_endpoint:
        sys.exit("ERROR: set BOTH AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT "
                 "(or unset both and set OPENAI_API_KEY).")
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        return OpenAI()
    sys.exit("ERROR: no LLM credentials found. Set AZURE_OPENAI_API_KEY and "
             "AZURE_OPENAI_ENDPOINT (Azure OpenAI), or OPENAI_API_KEY (OpenAI).")


def sanitize(s):
    """Replace curly quotes and unicode dashes so the JSON reply parses."""
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'")
             .replace("–", "-").replace("—", "-").replace("‐", "-"))


def load_celltype_metadata(csv_path):
    """'cluster:NNN' -> descriptive fields (Siletti et al. 2023 annotations).

    The CSV needs the columns cluster_id, Cluster name, Class auto-annotation,
    Supercluster, Neurotransmitter auto-annotation, Top three regions,
    Top Enriched Genes. Rows are disease-independent; the first row per cluster wins.
    """
    meta = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            cid = f"cluster:{r['cluster_id'].strip()}"
            meta.setdefault(cid, {
                "name": r["Cluster name"].strip(),
                "cls": r["Class auto-annotation"].strip() or "NA",
                "supercluster": r["Supercluster"].strip(),
                "nt": r["Neurotransmitter auto-annotation"].strip() or "NA",
                "regions": r["Top three regions"].strip(),
                "markers": r["Top Enriched Genes"].strip(),
            })
    return meta


def format_celltype_context(m):
    return (f"- Cell type: {m['name']}\n- Cell class: {m['cls']}\n"
            f"- Supercluster: {m['supercluster']}\n- Neurotransmitter identity: {m['nt']}\n"
            f"- Predominant brain regions: {m['regions']}\n"
            f"- Marker (top enriched) genes for this cell type: {m['markers']}")
