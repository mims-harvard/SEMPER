#!/usr/bin/env python3

import argparse
import gc
import glob
import json
import os
import re
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

TISSUE_ORDER = ["brain", "ovary", "lung", "liver", "pancreas",
                "blood", "breast", "throat", "colon", "stomach", "kidney"]

_ZWS = "\u200b\u200c\u200d\ufeff"

def _clean(s):
    if s is None: return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})

def parse_digits(s: str) -> str:
    m = re.search(r"(\d+)", str(s))
    return m.group(1) if m else ""

def _despine(ax):
    sns.despine(ax=ax)

def load_ensp_to_gene(json_path: str) -> dict:
    """
    Load ensp_to_gene.json and build a normalised ENSP → preferred gene symbol
    mapping (handles '9606.' prefix and version suffixes).
    """
    with open(json_path) as f:
        raw = json.load(f)
    norm = {}
    for k, v in raw.items():
        v = str(v).strip()
        k = str(k)
        bare = k.split("9606.", 1)[1] if k.startswith("9606.") else k
        bare = re.sub(r"\.\d+$", "", bare)
        norm[bare]          = v
        norm[k]             = v
        if bare.startswith("ENSP"):
            norm["9606." + bare] = v
    return norm

def load_embeddings_to_disk(embed_path, label_path, ensp_to_gene, work_dir):
    print("[INFO] Loading embeddings and serialising clusters to disk...")
    embed = torch.load(embed_path, map_location="cpu")

    with open(label_path) as f:
        labels = json.loads(f.read().replace("'", '"'))

    df = pd.DataFrame({"ensp": labels["Name"], "cell_type": labels["Cell Type"]})
    df = df[~df["cell_type"].astype(str).str.startswith(("CCI", "BTO", "Sanity"))].copy()
    df["cid_str"] = df["cell_type"].apply(lambda x: parse_digits(str(x)))
    df = df[df["cid_str"] != ""]

    unique_cids = sorted(df["cid_str"].unique())
    cid_to_rows = df.groupby("cid_str")

    os.makedirs(work_dir, exist_ok=True)
    cluster_index    = []
    all_genes_semper = set()

    for k in tqdm(range(len(embed)), desc="Mapping & saving clusters"):
        if k >= len(unique_cids): continue
        cid = unique_cids[k]
        if cid not in cid_to_rows.groups: continue
        sub    = cid_to_rows.get_group(cid)
        tensor = embed.get(k)
        if tensor is None or tensor.shape[0] != len(sub): continue

        ensp_list = sub["ensp"].tolist()
        gene_list = [ensp_to_gene.get(e, "") for e in ensp_list]
        keep_idx  = [i for i, g in enumerate(gene_list) if g]
        if len(keep_idx) < 2: continue

        t_filt = F.normalize(tensor[keep_idx].float(), p=2, dim=1)
        g_filt = [gene_list[i] for i in keep_idx]

        npy_path = os.path.join(work_dir, f"cluster_{cid}.npy")
        np.save(npy_path, t_filt.numpy())
        del t_filt, tensor

        cluster_index.append({"key": f"cluster:{cid}", "genes": g_filt, "npy": npy_path})
        all_genes_semper.update(g_filt)

    del embed
    gc.collect()
    print(f"[INFO] Saved {len(cluster_index)} clusters | "
          f"{len(all_genes_semper):,} unique gene symbols.")
    return cluster_index, all_genes_semper

def discover_association_files(assoc_dir: str) -> dict:
    pattern = os.path.join(assoc_dir, "cohorts_combined_*_tumor_avg_outer_prob.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Laman Trip CSV files found in {assoc_dir}.\n"
            "Expected pattern: cohorts_combined_<tissue>_tumor_avg_outer_prob.csv\n"
            "Download from: https://doi.org/10.1038/s41467-022-34544-2"
        )
    out = {}
    for fp in files:
        base   = os.path.basename(fp)
        tissue = re.search(r"cohorts_combined_(.+?)_tumor", base).group(1)
        out[tissue] = fp
    return out

def load_tissue_csv(fp: str, tissue: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    score_col = [c for c in df.columns if c not in ("prot1", "prot2")][0]
    df["g1"] = df[["prot1", "prot2"]].min(axis=1)
    df["g2"] = df[["prot1", "prot2"]].max(axis=1)
    df = (df[["g1", "g2", score_col]]
          .rename(columns={score_col: "assoc_score"})
          .drop_duplicates(subset=["g1", "g2"])
          .reset_index(drop=True))
    print(f"  Loaded '{tissue}': {len(df):,} pairs")
    return df

class PairLookup:
    """O(log N) vectorised lookup for (g1, g2) → float32 score."""
    def __init__(self, df: pd.DataFrame, vocab: dict):
        g1e  = df["g1"].map(vocab).to_numpy(dtype=np.uint32)
        g2e  = df["g2"].map(vocab).to_numpy(dtype=np.uint32)
        sc   = df["assoc_score"].to_numpy(dtype=np.float32)
        keys = g1e.astype(np.uint64) * (len(vocab) + 1) + g2e.astype(np.uint64)
        order       = np.argsort(keys, kind="stable")
        self.keys   = keys[order]
        self.scores = sc[order]
        self.vocab  = vocab
        self.nvocab = len(vocab)

    def query(self, g1_enc: np.ndarray, g2_enc: np.ndarray) -> np.ndarray:
        q   = g1_enc.astype(np.uint64) * (self.nvocab + 1) + g2_enc.astype(np.uint64)
        idx = np.clip(np.searchsorted(self.keys, q), 0, len(self.keys) - 1)
        out = np.full(len(q), np.nan, dtype=np.float32)
        hit = self.keys[idx] == q
        out[hit] = self.scores[idx[hit]]
        return out

def step1_correlate_all_tissues(
    cluster_index, all_genes_semper, tissue_files, vocab, out_prefix
) -> pd.DataFrame:
    all_results = []

    for tissue, fp in tissue_files.items():
        print(f"\n[Step 1] Correlating with tissue: {tissue}")
        assoc_df    = load_tissue_csv(fp, tissue)
        assoc_genes = set(assoc_df["g1"]) | set(assoc_df["g2"])

        # Encode and sort for fast lookup
        g1e   = assoc_df["g1"].map(vocab).to_numpy(dtype=np.uint32)
        g2e   = assoc_df["g2"].map(vocab).to_numpy(dtype=np.uint32)
        sc    = assoc_df["assoc_score"].to_numpy(dtype=np.float32)
        order = np.argsort(g1e, kind="stable")
        g1e, g2e, sc = g1e[order], g2e[order], sc[order]
        g1_unique, g1_counts = np.unique(g1e, return_counts=True)
        offsets = np.zeros(len(g1_unique) + 1, dtype=np.int64)
        np.cumsum(g1_counts, out=offsets[1:])
        g1_pos = {int(gid): (int(offsets[k]), int(offsets[k+1]))
                  for k, gid in enumerate(g1_unique)}

        for entry in tqdm(cluster_index, desc=f"  [{tissue}]", leave=False):
            ct_key = entry["key"]
            genes  = entry["genes"]
            ov_idx = [i for i, g in enumerate(genes) if g in assoc_genes]
            ov_g   = [genes[i] for i in ov_idx]
            if len(ov_g) < 5:
                all_results.append({"cluster": ct_key, "tissue": tissue,
                                     "n_pairs": 0, "spearman_r": np.nan})
                continue

            raw = np.load(entry["npy"], mmap_mode="r")
            sub = raw[ov_idx].astype(np.float32)
            del raw

            ct_enc     = np.array([vocab[g] for g in ov_g], dtype=np.uint32)
            ct_enc_set = set(ct_enc.tolist())
            g_to_local = {int(e): li for li, e in enumerate(ct_enc)}

            pi, pj, psc = [], [], []
            for li, g1id in enumerate(ct_enc):
                g1id = int(g1id)
                if g1id not in g1_pos: continue
                lo, hi = g1_pos[g1id]
                for k in range(lo, hi):
                    g2id = int(g2e[k])
                    if g2id in ct_enc_set and g2id != g1id:
                        lj = g_to_local[g2id]
                        if lj > li:
                            pi.append(li); pj.append(lj); psc.append(float(sc[k]))

            if len(psc) < 10:
                all_results.append({"cluster": ct_key, "tissue": tissue,
                                     "n_pairs": len(psc), "spearman_r": np.nan})
                del sub; continue

            sim_mat = sub @ sub.T
            ii, jj  = np.array(pi, dtype=np.int32), np.array(pj, dtype=np.int32)
            sims    = sim_mat[ii, jj]
            sr, _   = spearmanr(sims, np.array(psc, dtype=np.float32))
            all_results.append({"cluster": ct_key, "tissue": tissue,
                                 "n_pairs": len(psc), "spearman_r": float(sr)})
            del sub, sim_mat, ii, jj, sims, pi, pj, psc

        del assoc_df, g1e, g2e, sc, g1_pos
        gc.collect()

    df = pd.DataFrame(all_results)
    df.to_csv(f"{out_prefix}_step1_per_ct_correlations.csv", index=False)
    print(f"\n[SAVED] {out_prefix}_step1_per_ct_correlations.csv")
    return df

def step2_brain_enrichment(
    cluster_index, all_genes_semper, tissue_files, vocab,
    brain_threshold, out_prefix
) -> pd.DataFrame:
    print("\n[Step 2] Building brain-enrichment table (streaming per tissue)...")
    brain_df    = load_tissue_csv(tissue_files["brain"], "brain")
    assoc_genes = set(brain_df["g1"]) | set(brain_df["g2"])
    brain_lookup = PairLookup(brain_df, vocab)

    g1e_b = brain_df["g1"].map(vocab).to_numpy(dtype=np.uint32)
    g2e_b = brain_df["g2"].map(vocab).to_numpy(dtype=np.uint32)
    nvocab = len(vocab)
    keys_b = g1e_b.astype(np.uint64) * (nvocab + 1) + g2e_b.astype(np.uint64)
    bsc    = brain_df["assoc_score"].to_numpy(dtype=np.float64)

    other_sum   = np.zeros(len(brain_df), dtype=np.float64)
    other_count = np.zeros(len(brain_df), dtype=np.int32)

    for tissue, fp in tissue_files.items():
        if tissue == "brain": continue
        other = load_tissue_csv(fp, tissue)
        g1o   = other["g1"].map(vocab)
        g2o   = other["g2"].map(vocab)
        valid = g1o.notna() & g2o.notna()
        g1o   = g1o[valid].to_numpy(dtype=np.uint32)
        g2o   = g2o[valid].to_numpy(dtype=np.uint32)
        sco   = other["assoc_score"].to_numpy(dtype=np.float64)[valid.to_numpy()]
        ok    = g1o.astype(np.uint64) * (nvocab + 1) + g2o.astype(np.uint64)
        so    = np.argsort(ok, kind="stable")
        ok_s  = ok[so]; sco_s = sco[so]
        idx   = np.clip(np.searchsorted(ok_s, keys_b), 0, len(ok_s) - 1)
        hit   = ok_s[idx] == keys_b
        other_sum[hit]   += sco_s[idx[hit]]
        other_count[hit] += 1
        del other, g1o, g2o, sco, ok, so, ok_s, sco_s, idx, hit
        gc.collect()

    with np.errstate(invalid="ignore"):
        other_mean = np.where(other_count > 0,
                              other_sum / other_count, np.nan).astype(np.float32)
    bsc_f32  = bsc.astype(np.float32)
    enriched = (bsc_f32 > brain_threshold) & (bsc_f32 > other_mean)
    order_b  = np.argsort(keys_b, kind="stable")
    enr = {
        "keys":     keys_b[order_b],
        "enriched": enriched[order_b],
        "nvocab":   nvocab,
    }
    del brain_df, bsc, other_sum, other_count, bsc_f32, other_mean
    gc.collect()

    records = []
    for entry in tqdm(cluster_index, desc="  Brain enrichment per cluster"):
        genes  = entry["genes"]
        ov_idx = [i for i, g in enumerate(genes) if g in assoc_genes]
        ov_g   = [genes[i] for i in ov_idx]
        if len(ov_g) < 5: continue

        raw = np.load(entry["npy"], mmap_mode="r")
        sub = raw[ov_idx].astype(np.float32); del raw
        sim_mat = sub @ sub.T; del sub

        ii, jj   = np.triu_indices(len(ov_g), k=1)
        ga_str   = np.where(np.array(ov_g)[ii] <= np.array(ov_g)[jj],
                            np.array(ov_g)[ii], np.array(ov_g)[jj])
        gb_str   = np.where(np.array(ov_g)[ii] <= np.array(ov_g)[jj],
                            np.array(ov_g)[jj], np.array(ov_g)[ii])
        gae = np.array([vocab.get(g, 0) for g in ga_str], dtype=np.uint32)
        gbe = np.array([vocab.get(g, 0) for g in gb_str], dtype=np.uint32)

        bsc_pairs   = brain_lookup.query(gae, gbe)
        present     = ~np.isnan(bsc_pairs)
        if present.sum() < 10:
            del sim_mat, ii, jj; continue

        q   = gae[present].astype(np.uint64) * (nvocab+1) + gbe[present].astype(np.uint64)
        idx = np.clip(np.searchsorted(enr["keys"], q), 0, len(enr["keys"])-1)
        hit = enr["keys"][idx] == q
        enr_flags = np.zeros(present.sum(), dtype=bool)
        enr_flags[hit] = enr["enriched"][idx[hit]]

        sims_f   = sim_mat[ii[present], jj[present]].astype(np.float32)
        scores_f = bsc_pairs[present]
        r_all, _ = spearmanr(sims_f, scores_f)

        records.append({
            "cluster":          entry["key"],
            "n_pairs":          int(present.sum()),
            "n_enriched_pairs": int(enr_flags.sum()),
            "pct_enriched":     float(100 * enr_flags.mean()),
            "spearman_r_all":   float(r_all),
        })
        del sim_mat, ii, jj, gae, gbe, bsc_pairs, present, enr_flags, sims_f, scores_f
        gc.collect()

    df = pd.DataFrame(records)
    df.to_csv(f"{out_prefix}_step2_brain_enrichment.csv", index=False)
    print(f"[SAVED] {out_prefix}_step2_brain_enrichment.csv")
    return df

def compute_qvals(df_ref, tissues):
    pvals = []
    for tissue in tissues:
        if tissue == "brain":
            pvals.append(np.nan); continue
        bv = df_ref[df_ref["tissue"] == "brain"]["spearman_r"].dropna().values
        tv = df_ref[df_ref["tissue"] == tissue]["spearman_r"].dropna().values
        if len(bv) >= 5 and len(tv) >= 5:
            _, p = mannwhitneyu(bv, tv, alternative="greater")
            pvals.append(p)
        else:
            pvals.append(np.nan)
    valid = [not np.isnan(p) for p in pvals]
    vp    = [p for p in pvals if not np.isnan(p)]
    if vp:
        _, qc, _, _ = multipletests(vp, method="fdr_bh")
        qi = iter(qc)
        return [next(qi) if v else np.nan for v in valid]
    return pvals


def plot_tissue_comparison(df, tissues, q_vals, out_prefix, suffix):
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    df_clean   = df.dropna(subset=["spearman_r"])
    tissue_pos = {t: i for i, t in enumerate(tissues)}
    y_max      = df_clean["spearman_r"].quantile(0.99) * 1.10

    for idx, tissue in enumerate(tissues):
        tdf   = df_clean[df_clean["tissue"] == tissue]
        xp    = tissue_pos[tissue]
        color = "#2166ac" if tissue == "brain" else "#aaaaaa"
        if tdf.empty: continue
        ax.boxplot(tdf["spearman_r"].values, positions=[xp], widths=0.45,
                   patch_artist=True, manage_ticks=False,
                   flierprops=dict(marker="", markersize=0),
                   medianprops=dict(color="white", linewidth=1.5),
                   boxprops=dict(facecolor=color, alpha=0.5),
                   whiskerprops=dict(color=color, alpha=0.7),
                   capprops=dict(color=color, alpha=0.7))
        rng    = np.random.default_rng(xp)
        jitter = rng.uniform(-0.18, 0.18, size=len(tdf))
        ax.scatter(xp + jitter, tdf["spearman_r"].values,
                   color=color, alpha=0.4, s=10, zorder=4)
        if tissue != "brain":
            q = q_vals[idx]
            if not np.isnan(q):
                stars = ("***" if q < 0.001 else "**" if q < 0.01
                         else "*" if q < 0.05 else "ns")
                ax.text(xp, y_max, stars, ha="center", fontsize=10, color="#333333")

    ax.set_xticks(range(len(tissues)))
    ax.set_xticklabels([t.capitalize() for t in tissues], rotation=35, ha="right", fontsize=10)
    ax.set_xlabel("Tissue", fontsize=11)
    ax.set_ylabel("Spearman r\n(SEMPER cosine similarity vs. protein co-abundance score)",
                  fontsize=10)
    _despine(ax)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(f"{out_prefix}_{suffix}.{ext}", transparent=True, bbox_inches="tight")
        print(f"[PLOT] Saved {out_prefix}_{suffix}.{ext}")
    plt.close()


def step3_filter_and_plot(df_v3, df_v5, out_prefix, threshold=None):
    print("\n[Step 3] Filtering to brain-enriched clusters and plotting...")
    pct_col    = "pct_enriched"
    median_pct = float(df_v5[pct_col].median())
    threshold  = threshold if threshold is not None else median_pct
    print(f"  Median % enriched: {median_pct:.2f}%  |  Threshold: {threshold:.2f}%")

    high_clusters = set(df_v5[df_v5[pct_col] > threshold]["cluster"].values)
    print(f"  Clusters above threshold: {len(high_clusters)}/{len(df_v5)}")

    tissues = [t for t in TISSUE_ORDER if t in df_v3["tissue"].values]
    df_filt = df_v3[df_v3["cluster"].isin(high_clusters)].copy()

    q_all  = compute_qvals(df_v3,   tissues)
    q_filt = compute_qvals(df_filt, tissues)

    plot_tissue_comparison(df_v3,   tissues, q_all,  out_prefix, "all_clusters")
    plot_tissue_comparison(df_filt, tissues, q_filt, out_prefix, "enriched_clusters")

    rows = []
    for tissue in tissues:
        all_r  = df_v3[df_v3["tissue"]  == tissue]["spearman_r"].dropna()
        filt_r = df_filt[df_filt["tissue"] == tissue]["spearman_r"].dropna()
        rows.append({"tissue": tissue,
                     "median_spearman_all":      round(all_r.median(), 4),
                     "median_spearman_enriched": round(filt_r.median(), 4),
                     "n_all":      len(all_r),
                     "n_enriched": len(filt_r)})
    pd.DataFrame(rows).to_csv(f"{out_prefix}_step3_summary.csv", index=False)
    print(f"[SAVED] {out_prefix}_step3_summary.csv")


def main():
    ap = argparse.ArgumentParser(
        description="Fig 2f: SEMPER cosine similarity vs. Laman Trip co-abundance."
    )
    ap.add_argument("--embed-pt",          required=True)
    ap.add_argument("--labels",            required=True)
    ap.add_argument("--ensp-json",         required=True,
                    help="Path to ensp_to_gene.json")
    ap.add_argument("--assoc-dir",         required=True,
                    help="Directory containing Laman Trip co-abundance CSVs. "
                         "Download from: https://doi.org/10.1038/s41467-022-34544-2")
    ap.add_argument("--out-prefix",        required=True)
    ap.add_argument("--brain-threshold",   type=float, default=0.68,
                    help="Min brain score for a pair to be 'brain-enriched' "
                         "(default: 0.68, the 95th percentile used in the paper)")
    ap.add_argument("--enrich-threshold",  type=float, default=None,
                    help="Min %% brain-enriched pairs to keep a cluster in step 3 "
                         "(default: median across all clusters)")
    ap.add_argument("--work-dir",          default=None,
                    help="Temp directory for per-cluster .npy files "
                         "(default: <out-prefix>_workdir)")
    ap.add_argument("--keep-work-dir",     action="store_true",
                    help="Keep temp .npy files after run (useful for re-runs)")
    args = ap.parse_args()

    work_dir = args.work_dir or (args.out_prefix + "_workdir")

    # Load
    ensp_to_gene = load_ensp_to_gene(args.ensp_json)
    cluster_index, all_genes_semper = load_embeddings_to_disk(
        args.embed_pt, args.labels, ensp_to_gene, work_dir)
    del ensp_to_gene; gc.collect()

    print("\n[INFO] Discovering Laman Trip association atlases...")
    tissue_files = discover_association_files(args.assoc_dir)
    if "brain" not in tissue_files:
        raise RuntimeError("No brain atlas found — check --assoc-dir.")
    print(f"  Found: {', '.join(sorted(tissue_files))}")

    print("\n[INFO] Building global gene vocabulary...")
    all_atlas = set()
    for fp in tissue_files.values():
        gc_df = pd.read_csv(fp, usecols=["prot1", "prot2"])
        all_atlas.update(gc_df["prot1"]); all_atlas.update(gc_df["prot2"])
        del gc_df
    vocab = {g: i for i, g in enumerate(sorted(all_atlas | all_genes_semper))}
    del all_atlas; gc.collect()
    print(f"  Vocabulary size: {len(vocab):,}")

    df_v3 = step1_correlate_all_tissues(
        cluster_index, all_genes_semper, tissue_files, vocab, args.out_prefix)

    df_v5 = step2_brain_enrichment(
        cluster_index, all_genes_semper, tissue_files, vocab,
        args.brain_threshold, args.out_prefix)

    step3_filter_and_plot(df_v3, df_v5, args.out_prefix, args.enrich_threshold)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"\n[INFO] Temp work dir retained: {work_dir}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
