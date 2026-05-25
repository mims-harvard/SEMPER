import torch
import pickle
from Bio import SeqIO
from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

# Load the official model (600M variant)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = ESMC.from_pretrained("esmc_600m").to(device).eval()

def embed_sequence_official(seq: str):
    with torch.no_grad():
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein).to(device)
        output = client.logits(
            protein_tensor, 
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        
        embeddings = output.embeddings.squeeze(0)
        mean_embedding = embeddings[1:-1].mean(dim=0).cpu()
        
        del protein_tensor, output, embeddings
        return mean_embedding

seqs = SeqIO.to_dict(SeqIO.parse(
    f"protein_sequences.fasta",
    "fasta"
))

all_embeds = []
failed = []

for aaseq_id, record in tqdm(seqs.items()):
    try:
        # Using the official ESMC-600M embedding function
        all_embeds.append(embed_sequence_official(str(record.seq)))
    except Exception as e:
        print(f"Failed on {aaseq_id}: {e}")
        failed.append(aaseq_id)

all_embeds = torch.stack(all_embeds)

# Save results
torch.save(all_embeds, "protein_esmc_600m_official.pt")

with open("protein_esmc_600m_official.pkl", "wb") as fh:
    pickle.dump({seq_id: i for i, seq_id in enumerate(seqs.keys())}, fh)
