import os
import re
import json
import pickle
import random
import argparse
from ast import literal_eval
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="SEMPER multimodal contrastive alignment."
    )
    # Required paths
    parser.add_argument("--embed_path", required=True,
                        help="Per-cluster PPI-GT protein embeddings (.pth).")
    parser.add_argument("--label_path", required=True,
                        help="Label dictionary (.txt) from PPI-GT encoder.")
    parser.add_argument("--esm_pickle", required=True,
                        help="ESM index file (.pkl) from generate_ESMC_embeds.py.")
    parser.add_argument("--esm_embeds", required=True,
                        help="ESM embedding tensor (.pt) from generate_ESMC_embeds.py.")
    parser.add_argument("--save_dir", required=True,
                        help="Output directory for checkpoints and projections.")
    # Architecture
    parser.add_argument("--proj_dim", type=int, default=512,
                        help="Shared latent space dimensionality (default: 512).")
    parser.add_argument("--hidden_esm", type=int, nargs="+", default=[1024, 512],
                        help="Hidden layer sizes for ESM MLP (default: 1024 512).")
    parser.add_argument("--hidden_ppi", type=int, nargs="+", default=[512, 512],
                        help="Hidden layer sizes for PPI MLP (default: 512 512).")
    # Training
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size (default: 64).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for Adam (default: 1e-4).")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE loss temperature tau (default: 0.07).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    return parser.parse_args()


_ZWS = "\u200b\u200c\u200d\ufeff"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def clean(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().translate({ord(c): None for c in _ZWS})


def norm_ensp(x: str) -> str:
    """Normalize an ENSP ID: strip species prefix and version suffix."""
    s = clean(x)
    if s.startswith("9606."):
        s = s[5:]
    m = re.match(r"^(ENSP\d+)(?:\.\d+)?$", s)
    return m.group(1) if m else s


def load_labels(path: str) -> Dict:
    txt = open(path, "r").read()
    try:
        return json.loads(txt.replace("'", '"'))
    except Exception:
        return literal_eval(txt)

class MLP(nn.Module):
    """
    MLP projection head mapping a modality-specific embedding into the shared
    latent space. Final output is L2-normalized onto the unit hypersphere.
    Architecture: Linear -> ReLU (x n_hidden_layers) -> Linear -> L2-norm.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

def symmetric_infonce_loss(
    z_esm: torch.Tensor,
    z_ppi: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) loss over a minibatch of matched pairs.
    """
    logits = (z_esm @ z_ppi.T) / temperature          # [M, M]
    labels = torch.arange(z_esm.size(0), device=z_esm.device)
    loss_esm_to_ppi = F.cross_entropy(logits,   labels)
    loss_ppi_to_esm = F.cross_entropy(logits.T, labels)
    return (loss_esm_to_ppi + loss_ppi_to_esm) / 2.0

def build_perct_ppi_map(
    embed_path: str,
    label_dict: Dict,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, str], Dict[str, List[str]]]:
    """
    Parse the PPI-GT encoder output into a per-cluster protein embedding map.

    Returns:
        per_ct_ppi:           {ct_idx -> {ENSP_id -> tensor[D_ppi]}}
        idx_to_label:         {ct_idx -> 'cluster:N'}
        celltype_protein_dict:{cluster_label -> [ordered protein names]}
    """
    embed = torch.load(embed_path, map_location="cpu")

    # Build ct_idx -> cluster label mapping from the label dictionary
    celltypes = [c for c in label_dict["Cell Type"] if str(c).startswith("CCI")]
    idx_to_label: Dict[int, str] = {}
    for i, c in enumerate(celltypes):
        c = str(c)
        label = c.split("CCI_", 1)[1] if "CCI_" in c else (
            re.search(r"cluster:\d+", c).group(0)
            if re.search(r"cluster:\d+", c) else ""
        )
        if not label.startswith("cluster:"):
            m = re.search(r"(\d+)", label)
            label = f"cluster:{m.group(1)}" if m else ""
        idx_to_label[i] = label

    # Build cluster label -> ordered protein name list
    protein_names, protein_celltypes = [], []
    for c, p in zip(label_dict["Cell Type"], label_dict["Name"]):
        cs = str(c)
        if cs.startswith(("BTO", "CCI", "Sanity")):
            continue
        protein_names.append(p)
        protein_celltypes.append(cs)

    proteins = pd.DataFrame({"target": protein_names, "celltype": protein_celltypes})
    celltype_protein_dict: Dict[str, List[str]] = (
        proteins.pivot_table(values="target", index="celltype", aggfunc={"target": list})
        .to_dict()["target"]
    )

    # Map ct_idx -> {ENSP -> embedding vector}
    per_ct_ppi: Dict[int, Dict[str, torch.Tensor]] = {}
    for ct_idx, tensor in embed.items():
        if ct_idx not in idx_to_label:
            raise RuntimeError(f"ct_idx {ct_idx} not found in label mapping.")
        label = idx_to_label[ct_idx]
        prot_list = celltype_protein_dict.get(label)
        if prot_list is None:
            raise RuntimeError(f"No protein list for label '{label}'.")
        if tensor.shape[0] != len(prot_list):
            raise RuntimeError(
                f"Shape mismatch for '{label}': "
                f"tensor rows={tensor.shape[0]}, names={len(prot_list)}."
            )
        d: Dict[str, torch.Tensor] = {}
        for i in range(tensor.shape[0]):
            full_id = prot_list[i]
            ensp = norm_ensp(full_id.split(".", 1)[1] if "." in full_id else full_id)
            d[ensp] = tensor[i]
        per_ct_ppi[ct_idx] = d

    return per_ct_ppi, idx_to_label, celltype_protein_dict


def build_esm_map(esm_pickle: str, esm_embeds: str) -> Dict[str, torch.Tensor]:
    """
    Load ESM embeddings from generate_ESMC_embeds.py outputs.
    Returns {ENSP_id -> tensor[D_esm]}.
    """
    with open(esm_pickle, "rb") as f:
        idx_map = pickle.load(f)

    # Handle both id->index and index->id formats
    if isinstance(next(iter(idx_map.keys())), str):
        index_to_protein = {v: k for k, v in idx_map.items()}
    else:
        index_to_protein = idx_map

    E = torch.load(esm_embeds, map_location="cpu")
    esm_map: Dict[str, torch.Tensor] = {}
    for i in range(len(E)):
        prot = index_to_protein[i]
        ensp = norm_ensp(prot.split(".", 1)[-1] if "." in prot else prot)
        esm_map[ensp] = E[i].detach().cpu().float()
    return esm_map


def flatten_pairs(
    per_ct_ppi: Dict[int, Dict[str, torch.Tensor]],
    esm_map: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[int]]:
    """
    Build the flat list of matched (ESM, PPI) pairs across all cell types.
    Proteins without an ESM embedding are skipped here; they receive the
    mean-embedding fallback during final projection (build_esm_per_cluster).
    """
    esm_list, ppi_list, pids, cts = [], [], [], []
    missing = 0
    total = 0
    for ct_idx, ppi_dict in per_ct_ppi.items():
        for pid, ppi_vec in ppi_dict.items():
            total += 1
            v = esm_map.get(pid)
            if v is None:
                missing += 1
                continue
            esm_list.append(v)
            ppi_list.append(ppi_vec)
            pids.append(pid)
            cts.append(ct_idx)

    print(f"Contrastive training pairs: {len(pids):,}  "
          f"(no ESM coverage for {missing}/{total} = {missing/max(1,total):.1%})")
    return esm_list, ppi_list, pids, cts

def train_contrastive(
    esm_map: Dict[str, torch.Tensor],
    per_ct_ppi: Dict[int, Dict[str, torch.Tensor]],
    args,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train ESM and PPI MLP projection heads via symmetric InfoNCE loss.
    Both encoders are frozen; only the projection heads are optimized.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_esm = next(iter(esm_map.values())).numel()
    D_ppi = next(iter(next(iter(per_ct_ppi.values())).values())).numel()

    esm_proj = MLP(D_esm, args.hidden_esm, args.proj_dim).to(device)
    ppi_proj  = MLP(D_ppi, args.hidden_ppi, args.proj_dim).to(device)

    print(f"ESM projector: {count_trainable(esm_proj):,} trainable parameters")
    print(f"PPI projector: {count_trainable(ppi_proj):,} trainable parameters")
    print(f"Total:         {count_trainable(esm_proj) + count_trainable(ppi_proj):,} trainable parameters")

    optimizer = torch.optim.Adam(
        list(esm_proj.parameters()) + list(ppi_proj.parameters()), lr=args.lr
    )

    esm_list, ppi_list, _, _ = flatten_pairs(per_ct_ppi, esm_map)
    N = len(esm_list)
    if N == 0:
        raise RuntimeError(
            "No overlapping ESM/PPI proteins found. Check ID normalization."
        )

    for epoch in range(args.epochs):
        idxs = list(range(N))
        random.shuffle(idxs)
        running_loss = 0.0
        steps = 0
        for i in range(0, N, args.batch_size):
            batch_idx = idxs[i:i + args.batch_size]
            esm_batch = torch.stack([esm_list[j] for j in batch_idx]).to(device)
            ppi_batch = torch.stack([ppi_list[j] for j in batch_idx]).to(device)

            z_esm = esm_proj(esm_batch)
            z_ppi = ppi_proj(ppi_batch)

            loss = symmetric_infonce_loss(z_esm, z_ppi, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        print(f"[epoch {epoch + 1:03d}/{args.epochs}] loss = {running_loss / max(1, steps):.4f}")

    esm_proj.eval()
    ppi_proj.eval()
    return esm_proj, ppi_proj


def project_ppi_per_cluster(
    embed_path: str,
    label_dict: Dict,
    ppi_proj: nn.Module,
    idx_to_label: Dict[int, str],
    celltype_protein_dict: Dict[str, List[str]],
) -> Dict[int, torch.Tensor]:
    """Apply ppi_proj to each per-cluster raw tensor, preserving protein order."""
    raw = torch.load(embed_path, map_location="cpu")
    device = next(ppi_proj.parameters()).device
    out: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for ct_idx, tensor in raw.items():
            label = idx_to_label[ct_idx]
            prot_list = celltype_protein_dict[label]
            if tensor.shape[0] != len(prot_list):
                raise RuntimeError(f"Order/length mismatch for '{label}'.")
            rows = [ppi_proj(tensor[i].unsqueeze(0).to(device)).squeeze(0).cpu()
                    for i in range(tensor.shape[0])]
            out[ct_idx] = torch.stack(rows)
    return out


def build_esm_proj_dict(
    esm_map: Dict[str, torch.Tensor],
    esm_proj: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Project all ESM embeddings into the shared latent space."""
    device = next(esm_proj.parameters()).device
    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for pid, vec in esm_map.items():
            out[pid] = esm_proj(vec.to(device).unsqueeze(0)).squeeze(0).cpu()
    return out


def build_esm_per_cluster(
    esm_proj_dict: Dict[str, torch.Tensor],
    idx_to_label: Dict[int, str],
    celltype_protein_dict: Dict[str, List[str]],
) -> Dict[int, torch.Tensor]:
    """
    Construct per-cluster ESM tensors aligned to the PPI protein ordering.
    Proteins absent from the ESM database receive the mean projected embedding
    as a fallback (mu_esm in the paper).
    """
    fallback = torch.stack(list(esm_proj_dict.values())).mean(dim=0)

    out: Dict[int, torch.Tensor] = {}
    missing = 0
    total = 0
    for ct_idx, label in idx_to_label.items():
        rows = []
        for name in celltype_protein_dict[label]:
            ensp = norm_ensp(name.split(".", 1)[1] if "." in name else name)
            vec = esm_proj_dict.get(ensp)
            if vec is None:
                missing += 1
                vec = fallback
            rows.append(vec)
            total += 1
        out[ct_idx] = torch.stack(rows)

    print(f"ESM per-cluster fallback applied: {missing}/{total} = {missing/max(1,total):.1%}")
    return out


def concat_joint(
    ppi_proj_perct: Dict[int, torch.Tensor],
    esm_proj_perct: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Concatenate PPI and ESM projected embeddings per cluster."""
    out: Dict[int, torch.Tensor] = {}
    for ct_idx in ppi_proj_perct.keys():
        A = ppi_proj_perct[ct_idx]
        B = esm_proj_perct[ct_idx]
        if A.shape[0] != B.shape[0]:
            raise RuntimeError(f"Row mismatch at cluster idx {ct_idx}: "
                               f"PPI={A.shape[0]}, ESM={B.shape[0]}.")
        out[ct_idx] = torch.cat([A, B], dim=-1)
    return out

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading label dictionary...")
    labels = load_labels(args.label_path)

    print("Building per-cluster PPI embedding map...")
    per_ct_ppi, idx_to_label, celltype_protein_dict = build_perct_ppi_map(
        args.embed_path, labels
    )
    print(f"  Clusters: {len(per_ct_ppi)}  |  "
          f"Example labels: {sorted(set(idx_to_label.values()))[:6]}")

    print("Loading ESM embeddings...")
    esm_map = build_esm_map(args.esm_pickle, args.esm_embeds)
    print(f"  ESM proteins: {len(esm_map):,}  |  "
          f"D_esm = {next(iter(esm_map.values())).numel()}")

    print("Training contrastive projection heads...")
    esm_proj, ppi_proj = train_contrastive(esm_map, per_ct_ppi, args)

    D_esm = next(iter(esm_map.values())).numel()
    D_ppi = next(iter(next(iter(per_ct_ppi.values())).values())).numel()
    ckpt = {
        "esm_mlp_state": esm_proj.state_dict(),
        "ppi_mlp_state": ppi_proj.state_dict(),
        "esm_mlp_cfg": {"input_dim": D_esm, "hidden_dims": args.hidden_esm,
                        "output_dim": args.proj_dim},
        "ppi_mlp_cfg": {"input_dim": D_ppi, "hidden_dims": args.hidden_ppi,
                        "output_dim": args.proj_dim},
        "temperature": args.temperature,
        "seed": args.seed,
    }
    ckpt_path = os.path.join(args.save_dir, "clip_mlps.ckpt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved projection head checkpoint -> {ckpt_path}")

    print("Projecting PPI embeddings per cluster...")
    ppi_proj_perct = project_ppi_per_cluster(
        args.embed_path, labels, ppi_proj.to(device), idx_to_label, celltype_protein_dict
    )
    torch.save(ppi_proj_perct, os.path.join(args.save_dir, "ppi_projected.pth"))
    with open(os.path.join(args.save_dir, "ppi_projected_labels.txt"), "w") as f:
        f.write(str(labels))
    print(f"Saved ppi_projected.pth")

    print("Projecting ESM embeddings...")
    esm_proj_dict = build_esm_proj_dict(esm_map, esm_proj.to(device))
    ids_in_order  = sorted(esm_proj_dict.keys())
    Z_esm = torch.stack([esm_proj_dict[k] for k in ids_in_order])
    torch.save(Z_esm, os.path.join(args.save_dir, "esm_proj_embeddings.pt"))
    torch.save(esm_proj_dict, os.path.join(args.save_dir, "esm_proj_embeddings_idmap.pt"))
    with open(os.path.join(args.save_dir, "esm_proj_ids.json"), "w") as f:
        json.dump(ids_in_order, f)
    print(f"Saved esm_proj_embeddings.pt / esm_proj_embeddings_idmap.pt")

    print("Building ESM per-cluster tensors with fallback...")
    esm_proj_perct = build_esm_per_cluster(
        esm_proj_dict, idx_to_label, celltype_protein_dict
    )
    torch.save(esm_proj_perct, os.path.join(args.save_dir, "esm_projected.pth"))
    with open(os.path.join(args.save_dir, "esm_projected_labels.txt"), "w") as f:
        f.write(str(labels))
    print(f"Saved esm_projected.pth")

    print("Concatenating joint per-cluster tensors...")
    joint = concat_joint(ppi_proj_perct, esm_proj_perct)
    torch.save(joint, os.path.join(args.save_dir, "joint_projected.pth"))
    with open(os.path.join(args.save_dir, "joint_projected_labels.txt"), "w") as f:
        f.write(str(labels))
    print(f"Saved joint_projected.pth")

    any_ct = next(iter(ppi_proj_perct.keys()))
    print(
        f"\nExample cluster {any_ct}: "
        f"PPI {tuple(ppi_proj_perct[any_ct].shape)}  |  "
        f"ESM {tuple(esm_proj_perct[any_ct].shape)}  |  "
        f"Joint {tuple(joint[any_ct].shape)}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
