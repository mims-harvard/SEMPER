import argparse
import pickle

import torch
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mean-pooled ESM Cambrian embeddings for every sequence in a FASTA file."
    )
    parser.add_argument("--fasta", required=True,
                        help="Input FASTA (e.g. from fetch_sequences.py).")
    parser.add_argument("--out_embeds", required=True,
                        help="Output embedding tensor (.pt), shape [N_proteins, D].")
    parser.add_argument("--out_index", required=True,
                        help="Output index (.pkl) mapping protein ID -> row in --out_embeds.")
    parser.add_argument("--model", default="esmc_600m",
                        help="ESM-C checkpoint (default: esmc_600m, D = 1152).")
    return parser.parse_args()


def embed_sequence(client, seq: str, device):
    with torch.no_grad():
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein).to(device)
        output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )

        embeddings = output.embeddings.squeeze(0)
        mean_embedding = embeddings[1:-1].mean(dim=0).cpu()  # drop BOS/EOS tokens

        del protein_tensor, output, embeddings
        return mean_embedding


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = ESMC.from_pretrained(args.model).to(device).eval()

    seqs = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))

    all_embeds = []
    ids = []  # only successfully embedded sequences, so the index matches the tensor rows
    for aaseq_id, record in tqdm(seqs.items()):
        try:
            all_embeds.append(embed_sequence(client, str(record.seq), device))
            ids.append(aaseq_id)
        except Exception as e:
            print(f"Failed on {aaseq_id}: {e}")

    torch.save(torch.stack(all_embeds), args.out_embeds)
    with open(args.out_index, "wb") as fh:
        pickle.dump({seq_id: i for i, seq_id in enumerate(ids)}, fh)
    print(f"Embedded {len(ids)}/{len(seqs)} sequences -> {args.out_embeds}")


if __name__ == "__main__":
    main()
