#!/usr/bin/env python3
"""Cell type-aware Elo tournament with Gene Ontology evidence ("SEMPER+LLM" in Fig 6).

For every prompt:
  1. Take SEMPER's top-100 candidates (from semper_retrieval.py).
  2. Fetch each candidate's GO annotations from the Open Targets GraphQL API
     (HGNC -> ENSP -> ENSG), stringified and truncated to 1,500 characters.
  3. Seed Elo ratings linearly from SEMPER similarity into [1100, 1500].
  4. Play 5 Swiss-style rounds: sort by Elo (random tie order), pair adjacent
     candidates, and send up to 3 matches per request to the referee LLM, which
     sees the cell type description and both proteins' GO data and returns
     A / B / Tie (JSON mode). Elo update with K = 32, Tie = 0.5.
  5. Final ranking = Elo order.

GO terms are fetched with a direct GraphQL query. The ToolUniverse Open Targets
GO tool used earlier queries GeneOntologyTerm.name, which Open Targets renamed
to label, so it fails on every call.

    python llm_go_tournament.py \
        --input_json  inputs/prompts/generated_pd_messages_celltypecluster_zeroshotprompt_10prompt_t010.json \
        --cache_dir   outputs/retrieval/PD \
        --output_dir  outputs/go_tournament/PD \
        --disease     pd

Outputs per prompt (named after the retrieval cache file):
  tournament_GO_<file>          final Elo-ranked list (protein, similarity, elo, start_elo, matches_played)
  GOterms_cluster<N>_<file>     GO data shown to the referee (+ .txt with readable GO IDs)
  rationales_cluster<N>_<file>  per-match winner and one-sentence critique (+ .txt)
"""

import json
import os
import random
import re
import time
import urllib.request
from argparse import ArgumentParser

from common import (PAPER_MODEL, format_celltype_context, load_celltype_metadata,
                    make_client, resolve_disease)

HERE = os.path.dirname(os.path.abspath(__file__))
API_VERSION = "2025-04-01-preview"

# ==========================================
# CONFIGURATION (values used in the paper)
# ==========================================
BATCH_SIZE = 3
TOURNAMENT_ROUNDS = 5
K_FACTOR = 32
ELO_SEED_MIN = 1100
ELO_SEED_MAX = 1500
TOP_N_PARTICIPANTS = 100
GO_CHAR_LIMIT = 1500

# ==========================================
# Open Targets GraphQL (direct)
# ==========================================
OPENTARGETS_GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_GO_QUERY = """
query GO($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    approvedSymbol
    geneOntology {
      aspect
      evidence
      geneProduct
      source
      term { id label }
    }
  }
}
"""


def fetch_go_opentargets(ensembl_id, retries=4):
    """Return a list of {'id','label','aspect','source'} for one ENSG, or a
    short string on failure."""
    body = json.dumps({"query": _GO_QUERY, "variables": {"ensemblId": ensembl_id}}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OPENTARGETS_GQL_URL, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.loads(r.read())
            if payload.get("errors"):
                return f"OpenTargets query error: {payload['errors'][:1]}"
            tgt = (payload.get("data") or {}).get("target")
            if not tgt:
                return "No target found in OpenTargets."
            return [
                {"id": g["term"]["id"], "label": g["term"]["label"],
                 "aspect": g.get("aspect"), "source": g.get("source")}
                for g in (tgt.get("geneOntology") or [])
            ]
        except Exception as e:
            time.sleep(1.5 * (2 ** attempt) + random.random())
            last = e
    return f"No GO data available ({last})."


# ==========================================
# TOURNAMENT PROMPT & LOGIC
# ==========================================
TOURNAMENT_PROMPT_TEMPLATE = """
You are an expert biomedical researcher specializing in {disease}.
You are refereeing a "Relevance Tournament" for disease targets WITHIN A SPECIFIC brain cell type:
{ctx}

I will provide batches of 2 proteins (Protein A vs Protein B).
I have also provided their Gene Ontology (GO) annotations.
For each pair, decide: Which protein is a more functionally relevant and promising therapeutic target or biomarker for {disease} IN THIS SPECIFIC CELL TYPE -- taking its class, neurotransmitter identity, brain regions, and marker genes into account, not generic disease relevance?

### OUTPUT FORMAT (JSON ONLY):
{{
  "results": [
    {{
      "match_id": 0,
      "winner": "A",  // or "B" or "Tie"
      "critique": "Brief 1-sentence reason based on the provided GO data."
    }}
  ]
}}

### THE MATCHES AND GO DATA:
{matches_text}
"""


def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, actual_score_a, k=32):
    expected_a = calculate_expected_score(rating_a, rating_b)
    return rating_a + k * (actual_score_a - expected_a)


def extract_go_data(protein_name, go_data_dict):
    """Stringify one protein's GO data, truncated to protect the context window."""
    data = go_data_dict.get(protein_name, "No GO data available.")
    data_str = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
    return data_str[:GO_CHAR_LIMIT]


def extract_go_terms(go_data):
    """(GO_id, label) pairs from a GO blob, for the human-readable GO log."""
    found = []

    def walk(o):
        if isinstance(o, dict):
            gid = o.get("id") or o.get("goId") or o.get("go_id")
            label = o.get("term") or o.get("label") or o.get("name")
            if isinstance(gid, str) and gid.startswith("GO:"):
                found.append((gid, str(label) if label else ""))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and o.startswith("GO:"):
            found.append((o, ""))

    if isinstance(go_data, str):
        try:
            go_data = json.loads(go_data)
        except json.JSONDecodeError:
            pass
    walk(go_data)
    seen, out = set(), []
    for gid, lab in found:
        if gid not in seen:
            seen.add(gid)
            out.append((gid, lab))
    return out


def run_batch_matches(client, model, batch_pairs, go_data_dict, disease, ctx):
    matches_text = ""
    for i, (p_a, p_b, idx) in enumerate(batch_pairs):
        go_a = extract_go_data(p_a['protein'], go_data_dict)
        go_b = extract_go_data(p_b['protein'], go_data_dict)
        matches_text += f"MATCH_ID {i}: {p_a['protein']} vs {p_b['protein']}\n"
        matches_text += f"--- {p_a['protein']} GO Data ---\n{go_a}\n"
        matches_text += f"--- {p_b['protein']} GO Data ---\n{go_b}\n\n"

    prompt = TOURNAMENT_PROMPT_TEMPLATE.format(disease=disease, ctx=ctx, matches_text=matches_text)

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            if isinstance(data, dict):
                for key in ["matches", "results", "rankings"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                if "match_id" in data:
                    return [data]
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            wait_time = (2 * (2 ** attempt)) + (random.randint(0, 1000) / 1000)
            print(f"   API error (attempt {attempt+1}): {e}. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
    return []


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input_json", required=True, help="SEMPER prompts (seeds + 'celltype cluster:N').")
    parser.add_argument("--cache_dir", required=True, help="Output directory of semper_retrieval.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--disease", required=True,
                        help="pd, ad, ms, als or hd (uses the paper's disease names), or a free-text name.")
    parser.add_argument("--ensp_to_gene", default=os.path.join(HERE, "..", "data", "ensp_to_gene.json"),
                        help="ENSP -> HGNC map (default: evaluate/data/ensp_to_gene.json)")
    parser.add_argument("--ensp_to_ensg", default=os.path.join(HERE, "inputs", "ensp_to_ensg.json"),
                        help="ENSP -> ENSG map used to query Open Targets")
    parser.add_argument("--celltype_meta", default=os.path.join(HERE, "inputs", "celltype_metadata_top10.csv"))
    parser.add_argument("--model", default=PAPER_MODEL)
    parser.add_argument("--api_version", default=API_VERSION)
    parser.add_argument("--force_rerun", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--only_clusters", default=None,
                        help="Comma-separated cluster IDs to restrict to, e.g. '246'.")
    args = parser.parse_args()

    disease = resolve_disease(args.disease)
    meta = load_celltype_metadata(args.celltype_meta)
    only_clusters = ({c.strip() for c in args.only_clusters.split(",")}
                     if args.only_clusters else None)
    os.makedirs(args.output_dir, exist_ok=True)

    # HGNC -> ENSG via ENSP
    with open(args.ensp_to_ensg) as f:
        ensp_to_ensg = json.load(f)
    with open(args.ensp_to_gene) as f:
        ensp_to_pref = json.load(f)
    hgnc_to_ensg = {}
    for ensp_raw, hgnc_symbol in ensp_to_pref.items():
        ensg_id = ensp_to_ensg.get(ensp_raw.replace("9606.", ""))
        if ensg_id:
            hgnc_to_ensg[hgnc_symbol] = ensg_id

    with open(args.input_json) as f:
        prompts = json.load(f)
    client = make_client(args.api_version)

    for idx, prompt in enumerate(prompts):
        print(f"\nPrompt {idx+1}/{len(prompts)} for {disease}")
        matching_files = [f for f in os.listdir(args.cache_dir) if f.endswith(f"_{idx}.json")]
        if not matching_files:
            print(f"   No cached retrieval for idx {idx}")
            continue
        cached_file = matching_files[0]

        cluster_id = None
        m = (re.search(r"cluster:?(\d+)", cached_file) or
             re.search(r"cluster:?(\d+)", json.dumps(prompt)))
        if m:
            cluster_id = m.group(1)
        if only_clusters is not None and cluster_id not in only_clusters:
            continue

        stem = f"cluster{cluster_id}_" if cluster_id else ""
        output_path = os.path.join(args.output_dir, f"tournament_GO_{cached_file}")
        if not args.force_rerun and os.path.exists(output_path):
            print("   Already processed. Skipping.")
            continue

        with open(os.path.join(args.cache_dir, cached_file)) as f:
            protein_list = json.load(f)[:TOP_N_PARTICIPANTS]
        mc = meta.get(f"cluster:{cluster_id}")
        ctx = format_celltype_context(mc) if mc else "(cell-type metadata unavailable)"
        print(f"   {len(protein_list)} proteins (cluster {cluster_id}). Fetching GO data...")

        # --- GO data ---
        go_data_dict = {}
        for p in protein_list:
            ensg_id = hgnc_to_ensg.get(p["protein"])
            go_data_dict[p["protein"]] = (fetch_go_opentargets(ensg_id) if ensg_id
                                          else "No ENSG mapping found.")
        go_json_path = os.path.join(args.output_dir, f"GOterms_{stem}{cached_file}")
        with open(go_json_path, "w") as f:
            json.dump(go_data_dict, f, indent=2, default=str)
        with open(go_json_path.replace(".json", ".txt"), "w") as f:
            for prot, go in go_data_dict.items():
                terms = extract_go_terms(go)
                f.write(f"### {prot}\n")
                f.write("\n".join(f"  {gid}  {lab}" for gid, lab in terms) if terms else "  (none)")
                f.write("\n\n")

        # --- Elo seeding from SEMPER similarity ---
        sim_scores = [p.get('similarity', 0) for p in protein_list]
        min_sim = min(sim_scores) if sim_scores else 0
        max_sim = max(sim_scores) if sim_scores else 1
        sim_range = max_sim - min_sim if max_sim != min_sim else 1.0
        for p in protein_list:
            norm_score = (p.get('similarity', 0) - min_sim) / sim_range
            p['elo'] = ELO_SEED_MIN + (norm_score * (ELO_SEED_MAX - ELO_SEED_MIN))
            p['start_elo'] = p['elo']
            p['matches_played'] = 0

        rationale_log = []
        live_path = os.path.join(args.output_dir, f"rationales_{stem}{cached_file}".replace(".json", ".txt"))
        live = open(live_path, "w")
        live.write(f"# Rationales for {disease} | cluster {cluster_id} | {cached_file}\n\n")

        # --- Tournament ---
        for round_num in range(TOURNAMENT_ROUNDS):
            print(f"   Round {round_num + 1}/{TOURNAMENT_ROUNDS}...", end="", flush=True)
            random.shuffle(protein_list)
            protein_list.sort(key=lambda x: x['elo'], reverse=True)

            pairs = [(protein_list[i], protein_list[i + 1], i)
                     for i in range(0, len(protein_list) - 1, 2)]

            for b in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[b: b + BATCH_SIZE]
                batch_output = run_batch_matches(client, args.model, batch, go_data_dict, disease, ctx)

                for res in batch_output:
                    m_id = res.get("match_id")
                    winner = res.get("winner")
                    if isinstance(m_id, int) and m_id < len(batch):
                        p_a, p_b, _ = batch[m_id]
                        score_a = 1.0 if winner == "A" else (0.0 if winner == "B" else 0.5)

                        old_a, old_b = p_a['elo'], p_b['elo']
                        p_a['elo'] = update_elo(old_a, old_b, score_a, K_FACTOR)
                        p_b['elo'] = update_elo(old_b, old_a, 1.0 - score_a, K_FACTOR)
                        p_a['matches_played'] += 1
                        p_b['matches_played'] += 1

                        critique = res.get("critique", "")
                        rationale_log.append({
                            "round": round_num + 1,
                            "protein_a": p_a['protein'],
                            "protein_b": p_b['protein'],
                            "winner": winner,
                            "critique": critique,
                        })
                        live.write(f"[R{round_num+1}] {p_a['protein']} vs {p_b['protein']} "
                                   f"-> {winner}\n    {critique}\n")
                        live.flush()
            print(" done.")
        live.close()

        with open(os.path.join(args.output_dir, f"rationales_{stem}{cached_file}"), "w") as f:
            json.dump(rationale_log, f, indent=2)

        if sum(p['matches_played'] for p in protein_list) == 0:
            print("   No valid matches returned; not saving so the prompt is retried next run.")
            continue

        protein_list.sort(key=lambda x: x['elo'], reverse=True)
        with open(output_path, "w") as f:
            json.dump(protein_list, f, indent=2)
        print(f"   Saved {output_path} ({len(rationale_log)} rationales)")


if __name__ == "__main__":
    main()
