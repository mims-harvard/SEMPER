import argparse, csv, json, os, re, sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_ZWS = "\u200b\u200c\u200d\ufeff"

# ----------------- helpers -----------------

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

def parse_celltype_id_digits(label: str) -> str:
    s = _clean(label)
    m = re.match(r"^(\d+)", s) or re.search(r"cluster:(\d+)", s) or re.search(r"(\d+)", s)
    return m.group(1) if m else ""

def edgelist_to_cluster_label(s: str) -> str:
    d = parse_celltype_id_digits(s)
    return f"cluster:{d}" if d else ""

def normalize_ct_label(ct: str) -> str:
    """
    Normalize CT label strings to match your file naming convention:
    '101_edgelist.txt' -> '101_edgelist'
    """
    ct = _clean(ct)
    if ct.endswith(".txt"):
        ct = ct[:-4]
    return ct

def parse_member_celltypes(s: Optional[str]) -> Dict[str, str]:
    """
    Parse mapping like:
      'ENSP1@101_edgelist;9606.ENSP2@190_edgelist'
    -> {'ENSP1': '101_edgelist', 'ENSP2': '190_edgelist'}
    """
    out: Dict[str, str] = {}
    s = _clean(s)
    if _is_missing_token(s):
        return out
    for tok in s.split(";"):
        tok = _clean(tok)
        if not tok or "@" not in tok:
            continue
        ensp_raw, ct_raw = tok.split("@", 1)
        ensp = norm_ensp(ensp_raw)
        ct = normalize_ct_label(ct_raw)
        if ensp and ct:
            out[ensp] = ct
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

# ----------------- CT orbits loading (cached) -----------------

def load_node_map(node_map_path: str) -> Dict[int, str]:
    with open(node_map_path, "r") as f:
        node_map = json.load(f)
    return {int(k): v for k, v in node_map["idx_to_protein"].items()}

def load_orbits_for_ct(orbits_path: str, idx_to_protein: Dict[int, str]) -> pd.DataFrame:
    orbits = np.loadtxt(orbits_path)
    if orbits.ndim == 1:
        orbits = orbits.reshape(1, -1)
    n_nodes, n_orbits = orbits.shape

    proteins = []
    for i in range(n_nodes):
        if i not in idx_to_protein:
            raise ValueError(f"Index {i} missing from idx_to_protein.")
        proteins.append(idx_to_protein[i])

    norm_ids = [norm_ensp(p) for p in proteins]
    return pd.DataFrame(orbits, index=norm_ids, columns=[f"orbit_{i}" for i in range(n_orbits)])

class CTOrbitStore:
    """
    Lazy loader with cache:
      expects files in dir:
        <ct>_nodes.json
        <ct>_orbits.txt
    """
    def __init__(self, ct_orbits_dir: str, max_cache: int = 16):
        self.dir = ct_orbits_dir
        self.max_cache = max_cache
        self.cache: Dict[str, pd.DataFrame] = {}
        self._ct_exists = self._scan_available_cts(ct_orbits_dir)

    @staticmethod
    def _scan_available_cts(d: str) -> set:
        nodes = glob(os.path.join(d, "*_nodes.json"))
        out = set()
        for nj in nodes:
            base = os.path.basename(nj)
            ct = base.replace("_nodes.json", "")
            out.add(ct)
        return out

    def has_ct(self, ct: str) -> bool:
        return ct in self._ct_exists

    def get(self, ct: str) -> pd.DataFrame:
        if ct in self.cache:
            return self.cache[ct]

        if ct not in self._ct_exists:
            raise KeyError(ct)

        nodes_path = os.path.join(self.dir, f"{ct}_nodes.json")
        orbits_path = os.path.join(self.dir, f"{ct}_orbits.txt")
        if not os.path.exists(nodes_path) or not os.path.exists(orbits_path):
            raise KeyError(ct)

        idx_to_protein = load_node_map(nodes_path)
        orbit_df = load_orbits_for_ct(orbits_path, idx_to_protein)

        # cache eviction (simple FIFO)
        if self.max_cache > 0 and len(self.cache) >= self.max_cache:
            first_key = next(iter(self.cache.keys()))
            del self.cache[first_key]
        self.cache[ct] = orbit_df
        return orbit_df

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Zero-shot scoring using cell-type-specific graphlet orbit vectors.")
    ap.add_argument("--ct-orbits-dir", required=True,
                    help="Directory with <ct>_nodes.json and <ct>_orbits.txt files.")

    ap.add_argument("--eval-csv", required=True, help="Frozen eval CSV with label 0/1 and members_ensp.")
    ap.add_argument("--key-col", default="pos_key")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--celltype-col", default="cell_type")
    ap.add_argument("--members-col", default="members_ensp")
    ap.add_argument("--member-celltypes-col", default="member_celltypes")
    ap.add_argument("--k-col", default="k")

    ap.add_argument("--model-name", default="graphlets_ct")
    ap.add_argument("--score", choices=["pairwise_cos", "centroid_cos"], default="pairwise_cos")
    ap.add_argument("--min-proteins", type=int, default=3)

    ap.add_argument("--neg-missing-action", choices=["error", "drop", "use_row_ct"],
                    default="error",
                    help="If a negative has empty member_celltypes: error/drop/use_row_ct.")

    ap.add_argument("--max-ct-cache", type=int, default=16,
                    help="Max CT orbit tables to keep in RAM (0 disables caching).")

    ap.add_argument("--out-scores-csv", required=True)
    ap.add_argument("--out-metrics-json", required=True)
    args = ap.parse_args()

    score_fn = score_mean_pairwise_cosine if args.score == "pairwise_cos" else score_mean_cos_to_centroid

    df = pd.read_csv(args.eval_csv, dtype=str)
    for col in [args.key_col, args.label_col, args.celltype_col, args.members_col]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] eval-csv missing required column: {col}")

    # member_celltypes is only mandatory for negatives unless use_row_ct
    if args.neg_missing_action != "use_row_ct" and args.member_celltypes_col not in df.columns:
        raise SystemExit(f"[ERROR] eval-csv missing {args.member_celltypes_col} required for negatives.")

    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
    if df[args.label_col].isna().any():
        bad = df[df[args.label_col].isna()].head(5)
        raise SystemExit(f"[ERROR] some labels could not be parsed to numeric. Example rows:\n{bad}")

    df["__cluster_label__"] = df[args.celltype_col].apply(edgelist_to_cluster_label)

    store = CTOrbitStore(args.ct_orbits_dir, max_cache=args.max_ct_cache)

    skipped = {
        "missing_ct_file": 0,
        "missing_member_orbits": 0,
        "neg_missing_member_celltypes": 0,
        "neg_missing_member_mapping": 0,
        "too_small": 0,
    }
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
            row_ct = normalize_ct_label(ct_raw)
            cluster_label = r.get("__cluster_label__", "")

            members = split_members(r.get(args.members_col, ""))

            # determine which CT to use for each member
            member_to_ct: Dict[str, str] = {}
            if y == 1:
                # positive: all members use row CT
                member_to_ct = {m: row_ct for m in members}
            else:
                # negative: per-member CTs
                raw_mct = r.get(args.member_celltypes_col, "")
                m2c = parse_member_celltypes(raw_mct)

                if not m2c:
                    skipped["neg_missing_member_celltypes"] += 1
                    if args.neg_missing_action == "drop":
                        continue
                    if args.neg_missing_action == "use_row_ct":
                        member_to_ct = {m: row_ct for m in members}
                    else:
                        raise SystemExit(
                            f"[ERROR] Row {row_idx}: negative with empty/unparsable member_celltypes: {repr(raw_mct)}"
                        )
                else:
                    # require mapping for every member
                    missing_map = [m for m in members if m not in m2c]
                    if missing_map:
                        skipped["neg_missing_member_mapping"] += 1
                        raise SystemExit(
                            f"[ERROR] Row {row_idx}: {len(missing_map)} members lack mapping in member_celltypes; "
                            f"examples: {missing_map[:8]}"
                        )
                    member_to_ct = {m: m2c[m] for m in members}

            # fetch orbit vectors
            present_members, missing_members, vecs = [], [], []
            for m in members:
                ct = member_to_ct.get(m, row_ct)
                ct = normalize_ct_label(ct)

                if not store.has_ct(ct):
                    missing_members.append(m)
                    skipped["missing_ct_file"] += 1
                    continue

                orbit_df = store.get(ct)
                if m not in orbit_df.index:
                    missing_members.append(m)
                    skipped["missing_member_orbits"] += 1
                    continue

                present_members.append(m)
                vecs.append(orbit_df.loc[m].values.astype(np.float32, copy=False))

            if len(present_members) < args.min_proteins:
                skipped["too_small"] += 1
                continue

            X = np.stack(vecs, axis=0)
            sc = score_fn(X)

            k_declared = r.get(args.k_col, "")
            w.writerow([
                args.model_name, key, row_idx, y,
                ct_raw, cluster_label,
                k_declared, len(present_members),
                ";".join(present_members), ";".join(missing_members),
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
        "neg_missing_action": args.neg_missing_action,
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "skipped": skipped,
        "ct_orbits_dir": args.ct_orbits_dir,
    }

    with open(args.out_metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
