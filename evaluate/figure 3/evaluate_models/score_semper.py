import argparse, json, re, sys, csv
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_ZWS = "\u200b\u200c\u200d\ufeff"

def _clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def _is_missing_token(s: Optional[str]) -> bool:
    t = _clean(s).lower()
    return t == "" or t in {"na", "nan", "none", "null"}

def norm_ensp(x: str) -> str:
    s = _clean(x)
    if s.startswith("9606."):
        s = s[5:]
    m = re.match(r"^(ENSP\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else s

def parse_celltype_id_digits(label: str) -> str:
    s = _clean(label)
    m = re.match(r"^(\d+)", s) or re.search(r"cluster:(\d+)", s) or re.search(r"(\d+)", s)
    return m.group(1) if m else ""

def edgelist_to_cluster_label(s: str) -> str:
    d = parse_celltype_id_digits(s)
    return f"cluster:{d}" if d else ""

def split_members(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    vals: List[str] = []
    for t in str(cell).replace(",", ";").split(";"):
        t = _clean(t)
        if _is_missing_token(t):
            continue
        vals.append(norm_ensp(t))
    seen, out = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)

def score_mean_pairwise_cosine(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    Xn = l2_normalize_rows(X)
    S = Xn @ Xn.T
    iu = np.triu_indices(S.shape[0], k=1)
    return float(S[iu].mean())

def score_mean_cos_to_centroid(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    Xn = l2_normalize_rows(X)
    c = Xn.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return float((Xn @ c).mean())

# ----- labels helpers -----

def load_labels(label_dict_path: str) -> Dict[str, List[str]]:
    """
    Returns: celltype_protein_dict[label] = list of protein names (strings) aligned to embedding rows.
    """
    with open(label_dict_path, "r") as f:
        labels = json.loads(f.read().replace("'", '"'))

    protein_names, protein_celltypes = [], []
    for c, p in zip(labels["Cell Type"], labels["Name"]):
        cs = str(c)
        if cs.startswith(("BTO", "CCI", "Sanity")):
            continue
        protein_names.append(p)
        protein_celltypes.append(cs)

    proteins = pd.DataFrame.from_dict({"target": protein_names, "cell type": protein_celltypes})
    celltype_protein_dict = (
        proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list})
        .to_dict()["target"]
    )
    return celltype_protein_dict

def build_index_to_label_from_labels(labels_path: str) -> Dict[int, str]:
    with open(labels_path, "r") as f:
        labels = json.loads(f.read().replace("'", '"'))
    cci_rows = [c for c in labels["Cell Type"] if str(c).startswith("CCI")]
    index_to_label: Dict[int, str] = {}
    for idx, c in enumerate(cci_rows):
        c = str(c)
        if "CCI_" in c:
            label = c.split("CCI_", 1)[1]
        else:
            m = re.search(r"cluster:\d+", c)
            label = m.group(0) if m else f"cluster:{parse_celltype_id_digits(c)}"
        if not label.startswith("cluster:"):
            label = f"cluster:{parse_celltype_id_digits(label)}"
        index_to_label[idx] = label
    return index_to_label

def invert_index_to_label(index_to_label: Dict[int, str]) -> Dict[str, int]:
    return {v: k for k, v in index_to_label.items()}

def make_id_map_for_cluster(embed_tensor: torch.Tensor, prot_list: List[str]) -> Dict[str, np.ndarray]:
    if embed_tensor.shape[0] != len(prot_list):
        raise RuntimeError(f"Length mismatch: tensor rows={embed_tensor.shape[0]} names={len(prot_list)}")
    d: Dict[str, np.ndarray] = {}
    X = embed_tensor.detach().cpu().numpy()
    for i in range(X.shape[0]):
        full_id = str(prot_list[i])
        ensp = full_id.split(".", 1)[1] if "." in full_id else full_id
        ensp = norm_ensp(ensp)
        d[ensp] = X[i].astype(np.float32, copy=False)
    return d

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-csv", required=True, help="CSV with label 0/1, members, cell_type")
    ap.add_argument("--key-col", default="pos_key")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--members-col", default="members_ensp")
    ap.add_argument("--k-col", default="k")

    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-pt", required=True)
    ap.add_argument("--model-labels", required=True)

    ap.add_argument("--score", choices=["pairwise_cos", "centroid_cos"], default="pairwise_cos")
    ap.add_argument("--min-proteins", type=int, default=3)

    ap.add_argument("--out-scores-csv", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    score_fn = score_mean_pairwise_cosine if args.score == "pairwise_cos" else score_mean_cos_to_centroid

    df = pd.read_csv(args.eval_csv, dtype=str)

    for col in [args.key_col, args.label_col, args.celltype_col, args.members_col]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] eval-csv missing required column: {col}")

    # normalize label to int
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    if df[args.label_col].isna().any():
        bad = df[df[args.label_col].isna()].head(5)
        raise SystemExit(f"[ERROR] some labels could not be parsed to numeric. Example rows:\n{bad}")

    # Build cluster label
    df["__cluster_label__"] = df[args.celltype_col].apply(edgelist_to_cluster_label)

    celltype_prots = load_labels(args.model_labels)
    idx_to_label = build_index_to_label_from_labels(args.model_labels)
    label_to_idx = invert_index_to_label(idx_to_label)

    embed = torch.load(args.model_pt, map_location="cpu")

    # outputs
    skipped = {"bad_ct": 0, "too_small": 0, "no_cluster": 0}
    y_all, s_all = [], []

    with open(args.out_scores_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", args.key_col, "row_idx", "label",
            "cell_type_raw", "cluster_label",
            "k_declared", "k_present", "members_present", "members_missing",
            "score_model"
        ])

        for ct_label, sub in df.groupby("__cluster_label__", sort=True):
            if not isinstance(ct_label, str) or not ct_label.startswith("cluster:"):
                skipped["bad_ct"] += len(sub)
                continue
            if ct_label not in label_to_idx or ct_label not in celltype_prots:
                skipped["no_cluster"] += len(sub)
                continue

            m_idx = label_to_idx[ct_label]
            m_tensor = embed[m_idx]
            m_map = make_id_map_for_cluster(m_tensor, celltype_prots[ct_label])

            for row_idx, r in sub.iterrows():
                key = r.get(args.key_col, str(row_idx))
                y = int(float(r.get(args.label_col, "0")))
                ct_raw = r.get(args.celltype_col, "")
                members = split_members(r.get(args.members_col, ""))

                present = [m for m in members if m in m_map]
                missing = [m for m in members if m not in m_map]

                if len(present) < args.min_proteins:
                    skipped["too_small"] += 1
                    continue

                X = np.stack([m_map[m] for m in present], axis=0)
                sc = score_fn(X)

                k_declared = r.get(args.k_col, "")
                w.writerow([
                    args.model_name, key, row_idx, y,
                    ct_raw, ct_label,
                    k_declared, len(present),
                    ";".join(present), ";".join(missing),
                    sc
                ])

                y_all.append(y)
                s_all.append(sc)

            del m_map, m_tensor

    y = np.asarray(y_all, dtype=int)
    s = np.asarray(s_all, dtype=float)

    if len(y) == 0 or len(np.unique(y)) < 2:
        raise SystemExit("[ERROR] after filtering, not enough data / only one class remains.")

    metrics = {
        "model": args.model_name,
        "N_scored": int(len(y)),
        "score": args.score,
        "min_proteins": int(args.min_proteins),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "skipped": skipped,
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
