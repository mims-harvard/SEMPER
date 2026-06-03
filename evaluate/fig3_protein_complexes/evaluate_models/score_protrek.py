import argparse, json, re, sys, csv, gzip, pickle
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

# ----------------- prot-trek embedding loader -----------------

def load_dict_pickle_gz(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def strip_9606_prefix_keys(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    collisions = 0
    for k, v in d.items():
        kk = k
        if isinstance(kk, str) and kk.startswith("9606."):
            kk = kk[5:]
        kk = norm_ensp(kk)
        if kk in out and kk != k:
            collisions += 1
        out[kk] = v
    if collisions:
        print(f"[WARN] {collisions} key collisions after stripping prefix; last one wins.", file=sys.stderr)
    return out

def coerce_vec(v) -> np.ndarray:
    # handle torch tensors or numpy arrays
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)
    if v.ndim != 1:
        v = v.reshape(-1)
    return v.astype(np.float32, copy=False)

# ----------------- optional: cell-type protein universe from label_dict -----------------

def load_ct_universe_from_label_dict(label_dict_path: str) -> Dict[str, set]:
    """
    Returns: ct_to_ensps['cluster:10'] = set(ENSP...)
    Uses the same label_dict JSON format you’ve been using.
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

    proteins = pd.DataFrame({"target": protein_names, "cell type": protein_celltypes})
    ct_to_full = (
        proteins.pivot_table(values="target", index="cell type", aggfunc={"target": list})
        .to_dict()["target"]
    )

    out: Dict[str, set] = {}
    for ct, full_ids in ct_to_full.items():
        # ct is expected to already be like 'cluster:10' in your labels
        ct_label = str(ct)
        s = set()
        for full_id in full_ids:
            full_id = str(full_id)
            ensp = full_id.split(".", 1)[1] if "." in full_id else full_id
            s.add(norm_ensp(ensp))
        out[ct_label] = s
    return out

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--eval-csv", required=True, help="Frozen CSV with label 0/1 and members_ensp")
    ap.add_argument("--key-col", default="pos_key")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--members-col", default="members_ensp")
    ap.add_argument("--k-col", default="k")

    ap.add_argument("--model-name", default="protrek")
    ap.add_argument("--protrek-pkl-gz", required=True, help="ENSP->vec map (.pkl.gz)")

    ap.add_argument("--label-dict", default=None,
                    help="Optional label_dict JSON (same as SEMPER) to restrict proteins by cell type")
    ap.add_argument("--restrict-to-ct-universe", action="store_true",
                    help="If set, only score members that are in that cell type’s protein universe (from --label-dict).")

    ap.add_argument("--score", choices=["pairwise_cos", "centroid_cos"], default="pairwise_cos")
    ap.add_argument("--min-proteins", type=int, default=3)

    ap.add_argument("--out-scores-csv", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    if args.restrict_to_ct_universe and not args.label_dict:
        raise SystemExit("[ERROR] --restrict-to-ct-universe requires --label-dict")

    score_fn = score_mean_pairwise_cosine if args.score == "pairwise_cos" else score_mean_cos_to_centroid

    df = pd.read_csv(args.eval_csv, dtype=str)
    for col in [args.key_col, args.label_col, args.celltype_col, args.members_col]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] eval-csv missing required column: {col}")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    if df[args.label_col].isna().any():
        bad = df[df[args.label_col].isna()].head(5)
        raise SystemExit(f"[ERROR] some labels could not be parsed to numeric. Example rows:\n{bad}")

    df["__cluster_label__"] = df[args.celltype_col].apply(edgelist_to_cluster_label)

    print(f"[INFO] Loading ProtTrek embeddings: {args.protrek_pkl_gz}", file=sys.stderr)
    emb_raw = load_dict_pickle_gz(args.protrek_pkl_gz)
    emb_map = strip_9606_prefix_keys(emb_raw)

    # coerce vectors once (so scoring loop is fast)
    # if this map is huge and memory is tight, we can lazy-coerce in-loop instead.
    for k in list(emb_map.keys()):
        emb_map[k] = coerce_vec(emb_map[k])

    D = None
    for v in emb_map.values():
        D = int(v.shape[0])
        break
    if D is None:
        raise SystemExit("[ERROR] ProtTrek map is empty.")

    print(f"[INFO] ProtTrek loaded: {len(emb_map)} proteins, dim={D}", file=sys.stderr)

    ct_universe = None
    if args.restrict_to_ct_universe:
        print(f"[INFO] Loading cell-type universes from label_dict: {args.label_dict}", file=sys.stderr)
        ct_universe = load_ct_universe_from_label_dict(args.label_dict)
        print(f"[INFO] Loaded universes for {len(ct_universe)} cell types", file=sys.stderr)

    skipped = {"bad_ct": 0, "missing_ct_universe": 0, "too_small": 0}
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

        for ct_label, sub in df.groupby("__cluster_label__", sort=True):
            if not isinstance(ct_label, str) or not ct_label.startswith("cluster:"):
                skipped["bad_ct"] += len(sub)
                continue

            allowed_set = None
            if args.restrict_to_ct_universe:
                allowed_set = ct_universe.get(ct_label)
                if allowed_set is None:
                    skipped["missing_ct_universe"] += len(sub)
                    continue

            for row_idx, r in sub.iterrows():
                key = r.get(args.key_col, str(row_idx))
                y = int(float(r.get(args.label_col, "0")))
                ct_raw = r.get(args.celltype_col, "")
                members = split_members(r.get(args.members_col, ""))

                # Filter members to those with embeddings (and optionally in ct universe)
                present = []
                missing = []
                for m in members:
                    if m not in emb_map:
                        missing.append(m)
                        continue
                    if allowed_set is not None and m not in allowed_set:
                        missing.append(m)
                        continue
                    present.append(m)

                if len(present) < args.min_proteins:
                    skipped["too_small"] += 1
                    continue

                X = np.stack([emb_map[m] for m in present], axis=0)
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
        "restrict_to_ct_universe": bool(args.restrict_to_ct_universe),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "skipped": skipped,
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
