"""High-quality training with complete ground truth mapping."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth, iter_source_tsv
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker
from src.features import enrich_record_dict
from src.train import build_pair_dataset, train_and_compare_models

def main():
    paths = get_default_paths()

    print("[1/5] Loading 1,000 S1 entities and full ground truth...")
    s1_df = load_source_tsv(paths["train_source1"], "S1 Train", "S1-", nrows=1000)
    s1_ids = s1_df["entity_id"].tolist()
    s1_set = set(s1_ids)

    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(s1_set)]

    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()

    for _, row in gt_filtered.iterrows():
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

    total_true_links = sum(len(v) for v in gt_map.values())
    print(f"Sample S1 count: {len(s1_ids):,}, S1 with matches: {len(gt_map):,}, Total true links: {total_true_links:,}")
    print(f"Target IDs to find: S2={len(needed_s2):,}, S3={len(needed_s3):,}")

    # Full scan of S2 to find all needed true targets + 15,000 distractors
    print("[2/5] Scanning S2 for all target IDs + distractor pool...")
    s2_hits = []
    s2_distractors = []
    distractor_cap = 15000

    for chunk in iter_source_tsv(paths["train_source2"], chunksize=250000):
        m = chunk[chunk["entity_id"].isin(needed_s2)]
        if len(m) > 0:
            s2_hits.append(m)
        if len(s2_distractors) * 250000 < distractor_cap:
            d = chunk[~chunk["entity_id"].isin(needed_s2)].head(5000)
            s2_distractors.append(d)
        total_found = sum(len(h) for h in s2_hits)
        if total_found >= len(needed_s2):
            print(f"Found all {total_found:,} S2 targets early!")
            break

    s2_targets = pd.concat(s2_hits + s2_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Collected S2 target pool: {len(s2_targets):,} records ({sum(len(h) for h in s2_hits):,} true targets).")

    print("Scanning S3 for all target IDs + distractor pool...")
    s3_hits = []
    s3_distractors = []

    for chunk in iter_source_tsv(paths["train_source3"], chunksize=250000):
        m = chunk[chunk["entity_id"].isin(needed_s3)]
        if len(m) > 0:
            s3_hits.append(m)
        if len(s3_distractors) * 250000 < distractor_cap:
            d = chunk[~chunk["entity_id"].isin(needed_s3)].head(5000)
            s3_distractors.append(d)
        total_found = sum(len(h) for h in s3_hits)
        if total_found >= len(needed_s3):
            print(f"Found all {total_found:,} S3 targets early!")
            break

    s3_targets = pd.concat(s3_hits + s3_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Collected S3 target pool: {len(s3_targets):,} records ({sum(len(h) for h in s3_hits):,} true targets).")

    target_df = pd.concat([s2_targets, s3_targets], ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"Total target pool (True + Distractors): {len(target_df):,}")

    print("[3/5] Normalizing and generating candidates...")
    s1_df["name_core"] = [normalize_name(n)[1] for n in s1_df["business_name"]]
    s1_df["addr_norm"] = [normalize_address(a)[0] for a in s1_df["business_address"]]
    target_df["name_core"] = [normalize_name(n)[1] for n in target_df["business_name"]]
    target_df["addr_norm"] = [normalize_address(a)[0] for a in target_df["business_address"]]

    all_candidates = []
    for country in ["US", "India"]:
        q_c = s1_df[s1_df["country"] == country]
        t_c = target_df[target_df["country"] == country]
        if len(q_c) == 0 or len(t_c) == 0:
            continue
        blocker = CountryBlocker(
            country=country,
            name_top_k=25,
            addr_top_k=8,
            name_sim_threshold=0.10,
            addr_sim_threshold=0.18
        )
        blocker.fit(t_c)
        cands = blocker.query(q_c)
        all_candidates.extend(cands)

    print(f"Generated {len(all_candidates):,} candidate pairs across {len(s1_ids):,} S1 queries.")

    print("[4/5] Pre-enriching records and building pairwise feature dataset...")
    s1_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in s1_df.iterrows()}
    cand_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in target_df.iterrows()}

    pair_df = build_pair_dataset(s1_dict, cand_dict, all_candidates, gt_map)
    pos_count = int(pair_df["label"].sum())
    neg_count = len(pair_df) - pos_count
    print(f"Pairwise dataset: {len(pair_df):,} total pairs ({pos_count:,} POSITIVES, {neg_count:,} NEGATIVES).")

    print("[5/5] Performing GroupShuffleSplit (80% train, 20% validation) grouped on source1_entity_id...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(pair_df, pair_df["label"], groups=pair_df["source1_entity_id"]))

    train_df = pair_df.iloc[train_idx].copy()
    val_df = pair_df.iloc[val_idx].copy()

    val_s1_ids = list(set(val_df["source1_entity_id"]))
    print(f"Train groups: {len(set(train_df['source1_entity_id'])):,}, Val groups: {len(val_s1_ids):,}")

    models_dir = PROJECT_ROOT / "models"
    exp_summary = train_and_compare_models(train_df, val_df, val_s1_ids, gt_map, output_dir=models_dir)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY TABLE (HIGH-QUALITY TRUE TARGET SET):")
    print("="*60)
    print(exp_summary.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
