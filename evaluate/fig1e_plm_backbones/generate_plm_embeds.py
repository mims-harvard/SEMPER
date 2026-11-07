#!/usr/bin/env python3
"""
Mean-pooled ESM-2 and ESM-3 embeddings for every sequence in a FASTA file
(Figure 1e backbones). ESM-C 300M / 600M embeddings come from
semper/esm/generate_ESMC_embeds.py (--model esmc_300m / esmc_600m).

Outputs use the same format as generate_ESMC_embeds.py, so they drop straight
into semper/contrastive/train_contrastive_MLP.py and
evaluate/fig4_target_prioritization/run_evaluations/neuro_eval_esm_pklpt_ct_network.py:
  --out_embeds  .pt   tensor [N_proteins, D]
  --out_index   .pkl  {protein_id: row_index}

Pooling (as used for the paper):
  ESM-2 (fair-esm): last-layer residue embeddings, BOS/EOS dropped, mean over
      residues. Sequences longer than --max_res residues are split into
      consecutive windows and the window means are averaged weighted by length.
      fp16 on GPU.
  ESM-3 (esm3-sm-open-v1, EvolutionaryScale `esm` package): per-residue
      embeddings from forward_and_sample, token sequences longer than
      --max_tokens split into near-equal chunks, BOS/EOS dropped, mean over
      residues.

fair-esm (ESM-2) and EvolutionaryScale esm (ESM-3) both install a module named
`esm`, so run ESM-2 and ESM-3 in separate environments (see README.md). If the
EvolutionaryScale package was renamed to `esm3` to coexist with fair-esm, it is
picked up automatically.
"""
import argparse
import pickle
from typing import List, Tuple

import torch
from Bio import SeqIO
from tqdm import tqdm

ESM2_MODELS = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D",
               "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"]
ESM3_MODELS = ["esm3-sm-open-v1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mean-pooled ESM-2 / ESM-3 embeddings for every sequence in a FASTA file."
    )
    parser.add_argument("--fasta", required=True,
                        help="Input FASTA (e.g. from semper/esm/fetch_sequences.py).")
    parser.add_argument("--model", required=True, choices=ESM2_MODELS + ESM3_MODELS,
                        help="Backbone checkpoint.")
    parser.add_argument("--out_embeds", required=True,
                        help="Output embedding tensor (.pt), shape [N_proteins, D].")
    parser.add_argument("--out_index", required=True,
                        help="Output index (.pkl) mapping protein ID -> row in --out_embeds.")
    parser.add_argument("--max_res", type=int, default=1000,
                        help="ESM-2: residues per window for long sequences (default: 1000).")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="ESM-3: tokens per chunk including BOS/EOS (default: 2048).")
    return parser.parse_args()


# -----------------------------
# ESM-2 (fair-esm)
# -----------------------------
class ESM2Embedder:
    def __init__(self, model_name: str, device: torch.device, max_res: int):
        import esm  # fair-esm
        if not hasattr(esm, "pretrained") or not hasattr(esm.pretrained, model_name):
            raise SystemExit("[ERROR] fair-esm not found. Install it with `pip install fair-esm` "
                             "in an environment without the EvolutionaryScale `esm` package.")
        model, alphabet = getattr(esm.pretrained, model_name)()
        self.device = device
        self.model = model.to(device).eval()
        if device.type == "cuda":
            self.model = self.model.half()
        self.batch_converter = alphabet.get_batch_converter()
        self.max_res = max_res

    def _one_pass(self, seq: str) -> torch.Tensor:
        _, _, tokens = self.batch_converter([("protein", seq)])
        tokens = tokens.to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            out = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        reps = out["representations"][self.model.num_layers].squeeze(0)  # [L+2, D]
        return reps[1:-1].mean(dim=0).float().cpu()  # drop BOS/EOS

    @torch.inference_mode()
    def __call__(self, seq: str) -> torch.Tensor:
        if len(seq) <= self.max_res:
            return self._one_pass(seq)
        total, weight = None, 0.0
        for i in range(0, len(seq), self.max_res):
            chunk = seq[i:i + self.max_res]
            e = self._one_pass(chunk) * len(chunk)
            total = e if total is None else total + e
            weight += len(chunk)
        return total / weight


# -----------------------------
# ESM-3 (EvolutionaryScale esm)
# -----------------------------
def get_chunk_idxs(n_tokens: int, max_len: int) -> List[Tuple[int, int]]:
    """Split n_tokens into near-equal chunks of at most max_len (no tiny trailing chunk)."""
    n_pieces = (n_tokens + max_len - 1) // max_len
    lo = n_tokens // n_pieces
    n_hi = n_tokens % n_pieces
    lens = [lo + 1] * n_hi + [lo] * (n_pieces - n_hi)
    idxs, start = [], 0
    for L in lens:
        idxs.append((start, start + L))
        start += L
    return idxs


class ESM3Embedder:
    def __init__(self, model_name: str, device: torch.device, max_tokens: int):
        try:
            from esm3.models.esm3 import ESM3
            from esm3.sdk.api import ESMProtein, ESMProteinTensor, SamplingConfig
        except ImportError:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, ESMProteinTensor, SamplingConfig
        self.ESMProtein, self.ESMProteinTensor, self.SamplingConfig = ESMProtein, ESMProteinTensor, SamplingConfig
        self.device = device
        self.client = ESM3.from_pretrained(model_name, device=device).eval()
        self.max_tokens = max_tokens

    @torch.no_grad()
    def __call__(self, seq: str) -> torch.Tensor:
        full = self.client.encode(self.ESMProtein(sequence=seq)).to("cpu")
        outs = []
        for start, end in get_chunk_idxs(len(seq) + 2, self.max_tokens):  # +2 for BOS/EOS
            sub = self.ESMProteinTensor(sequence=full.sequence[start:end]).to(self.device)
            out = self.client.forward_and_sample(sub, self.SamplingConfig(return_per_residue_embeddings=True))
            outs.append(out.per_residue_embedding.detach().float().cpu())
        per_tok = torch.cat(outs, dim=0)
        return per_tok[1:-1].mean(dim=0)  # drop BOS/EOS


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model in ESM2_MODELS:
        embed = ESM2Embedder(args.model, device, args.max_res)
    else:
        embed = ESM3Embedder(args.model, device, args.max_tokens)

    all_embeds, ids = [], []  # only successfully embedded sequences, so the index matches the tensor rows
    n_seen = 0
    for rec in tqdm(SeqIO.parse(args.fasta, "fasta"), desc=args.model):
        n_seen += 1
        try:
            all_embeds.append(embed(str(rec.seq).strip()))
            ids.append(rec.id)
        except Exception as e:
            print(f"Failed on {rec.id}: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not all_embeds:
        raise SystemExit(f"[ERROR] No sequences embedded from {args.fasta}")
    E = torch.stack(all_embeds)
    torch.save(E, args.out_embeds)
    with open(args.out_index, "wb") as fh:
        pickle.dump({seq_id: i for i, seq_id in enumerate(ids)}, fh)
    print(f"Embedded {len(ids)}/{n_seen} sequences, D = {E.shape[1]} -> {args.out_embeds}")


if __name__ == "__main__":
    main()
