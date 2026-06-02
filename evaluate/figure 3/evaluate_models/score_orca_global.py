import argparse, csv, json, re, sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_ZWS = "\u200b\u200c\u200d\ufeff"

# ----------------- text/id helpers -----------------

def _clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def _is_missing_token(s: Optional[str]) -> bool:
    t = _clean(s).lower()
    return t == "" or t in {"na", "nan", "none", "null"}

def norm_ensp(x: str) -> str:
    """Canonicalize ENSP IDs: strip '9606.' and version suffix."""
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

# ----------------- scoring -----------------

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

# ----------------- load graphlet orbits -----------------

def load_node_map(node_map_path: str):
    with open(node_map_path, "r") as f:
        node_map = json.load(f)
    idx_to_protein = {int(k): v for k, v in node_map["idx_to_protein"].items()}
    return idx_to_protein

def load_orbits(orbits_path: str, idx_to_protein: Dict[int, str]) -> pd.DataFrame:
    """
    ORCA orbits: row i corresponds to node i (0-based).
    Returns DataFrame indexed by normalized ENSP.
    """
    orbits = np.loadtxt(orbits_path)
    n_nodes, n_orbits = orbits.shape

    proteins = []
    for i in range(n_nodes):
        if i not in idx_to_protein:
            raise ValueError(f"Index {i} missing from idx_to_protein.")
        proteins.append(idx_to_protein[i])

    norm_ids = [norm_ensp(p) for p in proteins]
    orbit_df = pd.DataFrame(orbits, index=norm_ids, columns=[f"orbit_{i}" for i in range(n_orbits)])
    return orbit_df

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Zero-shot scoring using graphlet orbit vectors.")
    ap.add_argument("--orbits", required=True, help="ORCA orbit output (txt)")
    ap.add_argument("--node-map", required=True, help="Node map JSON with idx_to_protein/protein_to_idx")

    ap.add_argument("--eval-csv", required=True, help="Frozen eval CSV with label 0/1 and members_ensp")
    ap.add_argument("--key-col", default="pos_key")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--members-col", default="members_ensp")
    ap.add_argument("--k-col", default="k")

    ap.add_argument("--model-name", default="graphlets_global")
    ap.add_argument("--score", choices=["pairwise_cos", "centroid_cos"], default="pairwise_cos")
    ap.add_argument("--min-proteins", type=int, default=3)

    ap.add_argument("--out-scores-csv", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    score_fn = score_mean_pairwise_cosine if args.score == "pairwise_cos" else score_mean_cos_to_centroid

    print(f"[INFO] Loading node-map: {args.node_map}", file=sys.stderr)
    idx_to_protein = load_node_map(args.node_map)
    print(f"[INFO] Loading orbits:   {args.orbits}", file=sys.stderr)
    orbit_df = load_orbits(args.orbits, idx_to_protein)
    print(f"[INFO] Orbits loaded: {orbit_df.shape[0]} proteins x {orbit_df.shape[1]} orbits", file=sys.stderr)

    df = pd.read_csv(args.eval_csv, dtype=str)
    for col in [args.key_col, args.label_col, args.celltype_col, args.members_col]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] eval-csv missing required column: {col}")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    if df[args.label_col].isna().any():
        bad = df[df[args.label_col].isna()].head(5)
        raise SystemExit(f"[ERROR] some labels could not be parsed to numeric. Example rows:\n{bad}")

    df["__cluster_label__"] = df[args.celltype_col].apply(edgelist_to_cluster_label)

    skipped = {"too_small": 0}
    y_all, s_all = [], []

    with open(args.out_scores_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", args.key_col, "row_idx", "label",
            "cell_type_raw", "cluster_label",
            "k_declared", "k_present",
            "members_present", "members_missing",
            "score_model"
        ])

        for row_idx, r in df.iterrows():
            key = r.get(args.key_col, str(row_idx))
            y = int(float(r.get(args.label_col, "0")))
            ct_raw = r.get(args.celltype_col, "")
            ct_label = r.get("__cluster_label__", "")

            members = split_members(r.get(args.members_col, ""))
            present, missing = [], []
            for m in members:
                if m in orbit_df.index:
                    present.append(m)
                else:
                    missing.append(m)

            if len(present) < args.min_proteins:
                skipped["too_small"] += 1
                continue

            X = orbit_df.loc[present].values.astype(np.float32, copy=False)
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
        "orbit_dim": int(orbit_df.shape[1]),
        "N_orbit_proteins": int(orbit_df.shape[0]),
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
