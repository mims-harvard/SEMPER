import json
import os
import random
import time
import asyncio
from argparse import ArgumentParser
from openai import AzureOpenAI
from tooluniverse.mcp_server import OpenTargets_get_target_gene_ontology_by_ensemblID

BATCH_SIZE = 3          
TOURNAMENT_ROUNDS = 5   
K_FACTOR = 32           
ELO_SEED_MIN = 1100     
ELO_SEED_MAX = 1500     
TOP_N_PARTICIPANTS = 100
MODEL_NAME = "gpt-4.1"

# Mapping files — passed as CLI arguments (see argparse below)
mapping_file_ensg = None  # set from args
mapping_file_pref = None  # set from args

TOURNAMENT_PROMPT_TEMPLATE = """
You are an expert biomedical researcher specializing in {disease}.
You are refereeing a "Relevance Tournament" for disease targets.

I will provide batches of 2 proteins (Protein A vs Protein B). 
I have also provided their Gene Ontology (GO) annotations.
For each pair, decide: Which protein is a more functionally relevant and promising therapeutic target or biomarker for {disease}?

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
    """Fetches GO data from our pre-compiled dictionary."""
    data = go_data_dict.get(protein_name, "No GO data available.")
    
    # Convert dict/list to string and truncate to save context window
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data)
    else:
        data_str = str(data)
        
    return data_str[:1500] 

def run_batch_matches(client, batch_pairs, go_data_dict, disease):
    matches_text = ""
    for i, (p_a, p_b, idx) in enumerate(batch_pairs):
        go_a = extract_go_data(p_a['protein'], go_data_dict)
        go_b = extract_go_data(p_b['protein'], go_data_dict)
        
        matches_text += f"MATCH_ID {i}: {p_a['protein']} vs {p_b['protein']}\n"
        matches_text += f"--- {p_a['protein']} GO Data ---\n{go_a}\n"
        matches_text += f"--- {p_b['protein']} GO Data ---\n{go_b}\n\n"

    prompt = TOURNAMENT_PROMPT_TEMPLATE.format(disease=disease, matches_text=matches_text)
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            if isinstance(data, dict):
                for key in ["matches", "results", "rankings"]:
                    if key in data and isinstance(data[key], list): return data[key]
                if "match_id" in data: return [data]
            if isinstance(data, list): return data
            return []
        except Exception as e:
            wait_time = (2 * (2 ** attempt)) + (random.randint(0, 1000) / 1000)
            print(f"   ⚠️ API Error (Attempt {attempt+1}): {e}. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            
    return []

# ==========================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to the JSON file with generated prompts")
    parser.add_argument("--cache_dir", required=True, help="Directory containing the cached base results")
    parser.add_argument("--output_dir", required=True, help="Directory to save final GO tournament results")
    parser.add_argument("--disease", required=True, help="Target disease context (e.g., 'Parkinson\\'s Disease')")
    parser.add_argument("--force_rerun", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--ensp_to_ensg", required=True,
                        help="Path to ENSP_to_ENSG.json (ENSP -> Ensembl gene ID map)")
    parser.add_argument("--ensp_to_pref", required=True,
                        help="Path to ensp_to_gene.json (ENSP -> preferred HGNC symbol map)")
    parser.add_argument("--azure_endpoint", default="https://api.openai.com",
                        help="Azure OpenAI endpoint (default: public OpenAI API)")
    parser.add_argument("--api_version",    default="2025-04-01-preview")
    args = parser.parse_args()

    global mapping_file_ensg, mapping_file_pref
    mapping_file_ensg = args.ensp_to_ensg
    mapping_file_pref = args.ensp_to_pref
    print(f"Starting Tool + Tournament Pipeline for {args.disease}...")

    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)

    print("Building HGNC -> ENSG mapping...")
    with open(mapping_file_ensg, "r") as f:
        ensp_to_ensg = json.load(f)

    with open(mapping_file_pref, "r") as f:
        ensp_to_pref = json.load(f)

    hgnc_to_ensg = {}
    for ensp_raw, hgnc_symbol in ensp_to_pref.items():
        ensp_clean = ensp_raw.replace("9606.", "")
        ensg_id = ensp_to_ensg.get(ensp_clean)
        if ensg_id:
            hgnc_to_ensg[hgnc_symbol] = ensg_id

    with open(args.input_json, "r") as f:
        prompts = json.load(f)

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=args.api_version
    )

    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx+1}/{len(prompts)} for {args.disease}")

        matching_files = [f for f in os.listdir(args.cache_dir) if f.endswith(f"_{idx}.json")]
        if not matching_files:
            print(f"   No cached file found for idx {idx}")
            continue

        cached_file = matching_files[0]
        cached_path = os.path.join(args.cache_dir, cached_file)
        output_path = os.path.join(args.output_dir, f"tournament_GO_{cached_file}")

        if not args.force_rerun and os.path.exists(output_path):
            print(f"   Already processed. Skipping.")
            continue

        with open(cached_path, "r") as f:
            protein_list = json.load(f)[:TOP_N_PARTICIPANTS] 

        print(f"   Loaded {len(protein_list)} proteins. Fetching GO Data via MCP Tool...")

        go_data_dict = {}
        for p in protein_list:
            protein_symbol = p["protein"]
            ensg_id = hgnc_to_ensg.get(protein_symbol)
            
            if not ensg_id:
                go_data_dict[protein_symbol] = "No ENSG mapping found."
                continue

            try:
                raw_go_data = asyncio.run(OpenTargets_get_target_gene_ontology_by_ensemblID.run({"ensemblId": ensg_id}))
                
                # Extract TextContent block to prevent JSON serialization crash
                if hasattr(raw_go_data, "content") and isinstance(raw_go_data.content, list) and len(raw_go_data.content) > 0:
                    extracted_text = raw_go_data.content[0].text
                    try:
                        go_data = json.loads(extracted_text)
                    except json.JSONDecodeError:
                        go_data = extracted_text 
                else:
                    go_data = str(raw_go_data)
                    
                go_data_dict[protein_symbol] = go_data
                
            except Exception as e:
                print(f"   Tool error for {protein_symbol}: {e}")
                go_data_dict[protein_symbol] = "No GO data available."

        print("   Initializing Elo Seeds...")
        sim_scores = [p.get('similarity', 0) for p in protein_list]
        min_sim = min(sim_scores) if sim_scores else 0
        max_sim = max(sim_scores) if sim_scores else 1
        sim_range = max_sim - min_sim if max_sim != min_sim else 1.0

        for p in protein_list:
            norm_score = (p.get('similarity', 0) - min_sim) / sim_range
            p['elo'] = ELO_SEED_MIN + (norm_score * (ELO_SEED_MAX - ELO_SEED_MIN))
            p['start_elo'] = p['elo']
            p['matches_played'] = 0

        for round_num in range(TOURNAMENT_ROUNDS):
            print(f"   Round {round_num + 1}/{TOURNAMENT_ROUNDS}...", end="", flush=True)
            
            random.shuffle(protein_list)
            protein_list.sort(key=lambda x: x['elo'], reverse=True)

            pairs = []
            for i in range(0, len(protein_list) - 1, 2):
                pairs.append((protein_list[i], protein_list[i+1], len(pairs)))

            for i in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[i : i + BATCH_SIZE]
                # Pass the pre-fetched dictionary instead of raw tool output
                batch_output = run_batch_matches(client, batch, go_data_dict, args.disease) 
                
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
            print(" Done.")

        # --- 6. Save Results ---
        total_matches_played = sum(p['matches_played'] for p in protein_list)
        if total_matches_played == 0:
            print(f"   API failed to return any valid matches. Not saving so it retries next time.")
            continue

        protein_list.sort(key=lambda x: x['elo'], reverse=True)
        with open(output_path, "w") as f:
            json.dump(protein_list, f, indent=2)
            
        print(f"   Saved Tournament Results to: {output_path}")

    print("All processing complete.")

if __name__ == "__main__":
    main()
