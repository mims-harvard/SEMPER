import re
import time
import argparse
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences for SEMPER's ESM encoder."
    )
    parser.add_argument("--protein_list", default="unique_proteins.txt",
                        help="Newline-delimited protein ID list.")
    parser.add_argument("--out_fasta", default="protein_sequences.fasta",
                        help="Output cleaned FASTA path.")
    parser.add_argument("--out_failed", default="failed_proteins.txt",
                        help="Output path for unresolved protein IDs.")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="IDs per API request (default: 50).")
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds between API calls (default: 0.2).")
    return parser.parse_args()

def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_token(s: str) -> str:
    """Normalize a raw protein ID token."""
    s = s.strip()
    s = s.replace("9606.", "")       # strip STRING species prefix
    s = s.split("|", 1)[0]          # drop cluster suffixes
    s = re.sub(r"\.\d+$", "", s)    # drop version suffix (.1, .2, ...)
    return s


def is_ensp(x: str) -> bool:
    return bool(re.match(r"^ENSP\d+$", x))


def is_uniprot_entry_name(x: str) -> bool:
    """UniProt entry names follow the pattern GENE_HUMAN (all uppercase)."""
    return x.endswith("_HUMAN") and bool(re.match(r"^[A-Z0-9]+_[A-Z0-9]+$", x))


def strip_stop_codons(seq: str) -> str:
    """Remove stop-codon characters (*) to prevent ESM tokenization errors."""
    return seq.replace("*", "")

def load_protein_ids(path: str):
    ensp_ids, uniprot_names, other = [], [], []
    with open(path) as f:
        for line in f:
            token = clean_token(line)
            if not token:
                continue
            if is_ensp(token):
                ensp_ids.append(token)
            elif is_uniprot_entry_name(token):
                uniprot_names.append(token)
            else:
                other.append(token)

    print(f"Loaded {len(ensp_ids)} ENSP IDs, "
          f"{len(uniprot_names)} UniProt entry names, "
          f"{len(other)} unrecognized IDs.")
    if other[:5]:
        print(f"  Example unrecognized: {other[:5]}")
    return ensp_ids, uniprot_names


ENSEMBL_URL  = "https://rest.ensembl.org/sequence/id"
ENSEMBL_HDRS = {"Content-Type": "application/json", "Accept": "application/json"}
ENSEMBL_PRM  = {"type": "protein"}


def fetch_ensp_sequences(ensp_ids, batch_size, sleep):
    """
    Retrieve amino acid sequences for ENSP IDs via the Ensembl REST API.
    Returns (ensp_seq dict, missing list).
    """
    ensp_seq = {}
    with requests.Session() as S:
        for batch in batchify(ensp_ids, batch_size):
            r = S.post(ENSEMBL_URL, params=ENSEMBL_PRM,
                       json={"ids": batch}, headers=ENSEMBL_HDRS)
            if r.status_code == 200:
                for rec in r.json():
                    pid, seq = rec.get("id"), rec.get("seq")
                    if pid and seq:
                        ensp_seq[pid] = seq
            else:
                # Batch failed — retry individually to salvage good IDs
                print(f"  [Ensembl] batch error {r.status_code}, retrying individually...")
                for pid in batch:
                    r2 = S.get(f"{ENSEMBL_URL}/{pid}",
                               params=ENSEMBL_PRM,
                               headers={"Accept": "text/plain"})
                    if r2.status_code == 200 and r2.text and not r2.text.startswith("{"):
                        ensp_seq[pid] = r2.text.strip()
                    time.sleep(sleep / 4)
            time.sleep(sleep)

    missing = [pid for pid in ensp_ids if pid not in ensp_seq]
    print(f"Ensembl: retrieved {len(ensp_seq)}/{len(ensp_ids)} sequences "
          f"({len(missing)} missing).")
    return ensp_seq, missing


ARCHIVE_URL = "https://rest.ensembl.org/archive/id"


def rescue_deprecated_ensp(missing, ensp_seq, batch_size, sleep):
    """
    Query the Ensembl Archive endpoint to map deprecated ENSP IDs to their
    current successors, then fetch sequences for the successor IDs.
    Returns updated ensp_seq and a rescued_map {old_ensp: (new_ensp, seq)}.
    """
    if not missing:
        return ensp_seq, {}

    print(f"Attempting archive rescue for {len(missing)} deprecated ENSP IDs...")

    old2new = {}
    with requests.Session() as S:
        for batch in batchify(missing, batch_size):
            r = S.post(ARCHIVE_URL, headers=ENSEMBL_HDRS, json={"id": batch})
            if r.status_code == 200:
                for rec in r.json():
                    old = rec.get("id")
                    new = rec.get("latest") or rec.get("new_id") or rec.get("stable_id") or old
                    new = re.sub(r"\.\d+$", "", new)  # strip version suffix
                    if old and new:
                        old2new[old] = new
            else:
                # Fallback: query individually
                for pid in batch:
                    r2 = S.get(f"{ARCHIVE_URL}/{pid}", headers=ENSEMBL_HDRS)
                    if r2.status_code == 200:
                        j = r2.json()
                        new = j.get("latest") or j.get("stable_id") or j.get("id")
                        if new:
                            old2new[pid] = re.sub(r"\.\d+$", "", new)
                    time.sleep(sleep / 4)
            time.sleep(sleep)

    # Fetch sequences for successor IDs not already in ensp_seq
    need_new = sorted(set(old2new.values()) - set(ensp_seq.keys()))
    with requests.Session() as S:
        for batch in batchify(need_new, batch_size):
            r = S.post(ENSEMBL_URL, params=ENSEMBL_PRM,
                       json={"ids": batch}, headers=ENSEMBL_HDRS)
            if r.status_code == 200:
                for rec in r.json():
                    pid, seq = rec.get("id"), rec.get("seq")
                    if pid and seq:
                        ensp_seq[pid] = seq
            time.sleep(sleep)

    # Build old -> (new, seq) rescue map
    rescued = {}
    for old, new in old2new.items():
        seq = ensp_seq.get(new)
        if seq:
            rescued[old] = (new, seq)

    print(f"Archive rescue: recovered {len(rescued)}/{len(missing)} IDs.")
    return ensp_seq, rescued

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FASTA  = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def fetch_uniprot_sequences(uniprot_names, batch_size, sleep):
    """
    Map UniProt entry names (e.g. BRCA1_HUMAN) to accessions via the
    UniProtKB search endpoint, then download FASTA sequences per accession.
    Returns (uniprot_seq dict {accession: fasta_text}, failed list).
    """
    # Step 1: entry name -> accession
    name_to_acc = {}
    with requests.Session() as S:
        for batch in batchify(uniprot_names, 100):
            query = " OR ".join(f"id:{x}" for x in batch)
            r = S.get(UNIPROT_SEARCH,
                      params={"query": query, "format": "tsv",
                              "fields": "accession,id", "size": 500},
                      headers={"Accept": "text/tab-separated-values"})
            if r.status_code == 200:
                for line in r.text.strip().splitlines()[1:]:  # skip header
                    parts = line.split("\t")
                    if len(parts) == 2:
                        acc, entry = parts
                        name_to_acc[entry] = acc
            else:
                print(f"  [UniProt search] error {r.status_code}")
            time.sleep(sleep)

    print(f"UniProt: mapped {len(name_to_acc)}/{len(uniprot_names)} entry names to accessions.")

    # Step 2: accession -> FASTA
    uniprot_seq = {}
    failed_acc  = []
    with requests.Session() as S:
        for acc in name_to_acc.values():
            r = S.get(UNIPROT_FASTA.format(acc=acc),
                      headers={"Accept": "text/plain"})
            txt = (r.text or "").strip()
            if r.status_code == 200 and txt.startswith(">"):
                uniprot_seq[acc] = txt
            else:
                failed_acc.append(acc)
            time.sleep(sleep / 2)

    failed_names = [n for n, a in name_to_acc.items() if a in failed_acc]
    print(f"UniProt: retrieved {len(uniprot_seq)}/{len(name_to_acc)} FASTA sequences.")
    return uniprot_seq, failed_names


def write_fasta(out_path, ensp_ids, ensp_seq, rescued, uniprot_seq):
    """
    Write a single combined FASTA with stop codons stripped.
    ENSP IDs mapped via the archive fallback are annotated with their
    successor ID in the header for full traceability.
    """
    written = 0
    with open(out_path, "w") as out:
        # ENSP sequences (primary + archive-rescued)
        for old in ensp_ids:
            if old in ensp_seq:
                seq = strip_stop_codons(ensp_seq[old])
                out.write(f">{old}\n{seq}\n")
                written += 1
            elif old in rescued:
                new, seq = rescued[old]
                seq = strip_stop_codons(seq)
                out.write(f">{old}|mapped_to={new}\n{seq}\n")
                written += 1

        # UniProt sequences (already in FASTA format with headers)
        for fasta_txt in uniprot_seq.values():
            records = list(SeqIO.parse(
                __import__("io").StringIO(fasta_txt), "fasta"
            ))
            for record in records:
                record.seq = Seq(strip_stop_codons(str(record.seq)))
                SeqIO.write(record, out, "fasta")
                written += 1

    print(f"Wrote {written} sequences to {out_path}.")
    return written


def main():
    args = parse_args()

    # 1. Load and stratify
    ensp_ids, uniprot_names = load_protein_ids(args.protein_list)

    # 2. Ensembl primary retrieval
    ensp_seq, missing = fetch_ensp_sequences(
        ensp_ids, args.batch_size, args.sleep
    )

    # 3. Archive fallback for deprecated IDs
    ensp_seq, rescued = rescue_deprecated_ensp(
        missing, ensp_seq, args.batch_size, args.sleep
    )

    # 4. UniProt retrieval
    uniprot_seq, failed_uniprot = fetch_uniprot_sequences(
        uniprot_names, args.batch_size, args.sleep
    )

    # 5. Write combined FASTA
    write_fasta(args.out_fasta, ensp_ids, ensp_seq, rescued, uniprot_seq)

    # 6. Report and save failures
    failed_ensp = [
        pid for pid in ensp_ids
        if pid not in ensp_seq and pid not in rescued
    ]
    all_failed = failed_ensp + failed_uniprot

    if all_failed:
        with open(args.out_failed, "w") as f:
            f.write("\n".join(all_failed) + "\n")
        print(f"{len(all_failed)} proteins could not be retrieved. "
              f"Saved to {args.out_failed}.")
        print("  Note: the contrastive alignment stage uses a mean-embedding "
              "fallback for proteins absent from the sequence database.")
    else:
        print("All proteins successfully retrieved.")


if __name__ == "__main__":
    main()
