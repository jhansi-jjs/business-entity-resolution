"""Test multi-strategy candidate recall on sample."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
import numpy as np

from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth, iter_source_tsv
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker, evaluate_blocking

def main():
    paths = get_default_paths()

    s1_sample = load_source_tsv(paths["train_source1"], "S1 Train", "S1-", nrows=10000)
    gt_df = load_ground_truth(paths["train_gt"], nrows=10000)

    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            m_set = {m.strip() for m in raw_m.split(",") if m.strip()}
            gt_map[s1] = m_set
            for mid in m_set:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

    eval_s1_ids = [s1 for s1 in s1_sample["entity_id"] if s1 in gt_map][:25]
    eval_gt_map = {s1: gt_map[s1] for s1 in eval_s1_ids}
    total_eval_links = sum(len(v) for v in eval_gt_map.values())
    
    needed_eval_s2 = {mid for mids in eval_gt_map.values() for mid in mids if mid.startswith("S2-")}
    needed_eval_s3 = {mid for mids in eval_gt_map.values() for mid in mids if mid.startswith("S3-")}

    # Get targets + 50k distractors
    s2_rows = []
    distractors_s2 = 0
    for chunk in iter_source_tsv(paths["train_source2"], chunksize=100000):
        hit = chunk[chunk["entity_id"].isin(needed_eval_s2)]
        if len(hit) > 0:
            s2_rows.append(hit)
        if distractors_s2 < 30000:
            take = chunk[~chunk["entity_id"].isin(needed_eval_s2)].head(30000 - distractors_s2)
            s2_rows.append(take)
            distractors_s2 += len(take)
        found_s2 = sum(len(h) for h in s2_rows) - distractors_s2
        if found_s2 >= len(needed_eval_s2) and distractors_s2 >= 30000:
            break

    s3_rows = []
    distractors_s3 = 0
    for chunk in iter_source_tsv(paths["train_source3"], chunksize=100000):
        hit = chunk[chunk["entity_id"].isin(needed_eval_s3)]
        if len(hit) > 0:
            s3_rows.append(hit)
        if distractors_s3 < 30000:
            take = chunk[~chunk["entity_id"].isin(needed_eval_s3)].head(30000 - distractors_s3)
            s3_rows.append(take)
            distractors_s3 += len(take)
        found_s3 = sum(len(h) for h in s3_rows) - distractors_s3
        if found_s3 >= len(needed_eval_s3) and distractors_s3 >= 30000:
            break

    target_pool = pd.concat(s2_rows + s3_rows, ignore_index=True).drop_duplicates(subset=["entity_id"])
    eval_s1_df = s1_sample[s1_sample["entity_id"].isin(set(eval_s1_ids))].copy()

    # Pre-normalize
    eval_s1_df["name_core"] = [normalize_name(n)[1] for n in eval_s1_df["business_name"]]
    eval_s1_df["addr_norm"] = [normalize_address(a)[0] for a in eval_s1_df["business_address"]]
    target_pool["name_core"] = [normalize_name(n)[1] for n in target_pool["business_name"]]
    target_pool["addr_norm"] = [normalize_address(a)[0] for a in target_pool["business_address"]]

    all_candidates = []
    for country in ["US", "India"]:
        q_c = eval_s1_df[eval_s1_df["country"] == country]
        t_c = target_pool[target_pool["country"] == country]
        if len(q_c) == 0 or len(t_c) == 0:
            continue
        blocker = CountryBlocker(
            country=country,
            name_top_k=30,
            addr_top_k=10,
            name_sim_threshold=0.10,
            addr_sim_threshold=0.20
        )
        blocker.fit(t_c)
        cands = blocker.query(q_c)
        all_candidates.extend(cands)

    metrics = evaluate_blocking(all_candidates, eval_gt_map, eval_s1_ids)
    print("\n" + "="*50)
    print("MULTI-STRATEGY BLOCKING RECALL RESULTS:")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
