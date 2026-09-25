"""Benchmark script to evaluate blocking recall on training ground truth."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
import numpy as np

from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth
from src.normalize import normalize_name
from src.blocking import CountryBlocker, evaluate_blocking

def main():
    paths = get_default_paths()

    print("Loading sample of training data for blocking benchmark (15,000 S1 queries)...")
    s1_df = load_source_tsv(paths["train_source1"], "S1 Train", "S1-", nrows=15000)
    
    # Get ground truth for these 15,000 entities
    s1_ids = s1_df["entity_id"].tolist()
    s1_set = set(s1_ids)
    
    gt_df = load_ground_truth(paths["train_gt"], nrows=50000)
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(s1_set)]
    
    gt_map = {}
    needed_target_ids = set()
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            m_set = {m.strip() for m in raw_m.split(",") if m.strip()}
            gt_map[s1] = m_set
            needed_target_ids.update(m_set)

    print(f"Sample queries: {len(s1_ids):,}, S1 with true matches in sample: {len(gt_map):,}, Total true links in sample: {sum(len(v) for v in gt_map.values()):,}")

    # Load targets: load 150,000 records from S2 and S3 + any target records known to match our sample
    print("Loading target records (Source 2 and Source 3)...")
    s2_df = load_source_tsv(paths["train_source2"], "S2 Train", "S2-", nrows=150000)
    s3_df = load_source_tsv(paths["train_source3"], "S3 Train", "S3-", nrows=150000)

    # Ensure all needed ground truth target IDs are present in targets pool
    present_targets = set(s2_df["entity_id"]).union(set(s3_df["entity_id"]))
    missing_gt_targets = needed_target_ids - present_targets
    print(f"Needed GT targets: {len(needed_target_ids):,}. Already in 300k sample: {len(needed_target_ids - missing_gt_targets):,}")

    # For fair evaluation of recall, we evaluate on S1 queries whose targets are in the target pool
    eval_s1_ids = [s1 for s1 in s1_ids if not (gt_map.get(s1, set()) - present_targets)]
    print(f"S1 queries with all targets present in index pool: {len(eval_s1_ids):,}")

    eval_s1_df = s1_df[s1_df["entity_id"].isin(set(eval_s1_ids))].copy()

    # Pre-normalize name cores
    print("Normalizing name cores...")
    eval_s1_df["name_core"] = [normalize_name(n)[1] for n in eval_s1_df["business_name"]]
    
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    targets_df["name_core"] = [normalize_name(n)[1] for n in targets_df["business_name"]]

    print("\n--- Running Blocking for US and India ---")
    all_candidates = []
    for country in ["US", "India"]:
        q_country = eval_s1_df[eval_s1_df["country"] == country]
        t_country = targets_df[targets_df["country"] == country]
        print(f"Country {country}: {len(q_country):,} queries vs {len(t_country):,} target candidates")
        if len(q_country) == 0 or len(t_country) == 0:
            continue

        blocker = CountryBlocker(country=country, top_k=20, sim_threshold=0.20)
        blocker.fit(t_country)
        cands = blocker.query(q_country)
        print(f"  -> Generated {len(cands):,} candidate pairs for {country}")
        all_candidates.extend(cands)

    metrics = evaluate_blocking(all_candidates, gt_map, eval_s1_ids)
    print("\n" + "="*50)
    print("BLOCKING EVALUATION METRICS:")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
