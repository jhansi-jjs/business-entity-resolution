"""Create reproducible stratified train, tune, and assessment splits for S1 queries."""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth


def create_splits(
    sample_size: int = 2500,
    seed: int = 42,
    output_dir: Path = None
):
    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = get_default_paths()
    print("Loading Ground Truth...")
    gt_df = load_ground_truth(paths["train_gt"])
    print(f"Total GT rows: {len(gt_df):,}")

    # Fast vectorized map of S1 to match count
    raw_m = gt_df["matched_entity_ids"].fillna("").astype(str)
    counts = raw_m.apply(lambda x: len([m for m in x.split(",") if m.strip()]) if x and x != "nan" else 0)
    s1_match_counts = dict(zip(gt_df["source1_entity_id"].astype(str), counts))
    print(f"Loaded match counts for {len(s1_match_counts):,} S1 entities.")

    # Load first 20,000 S1 records to sample from
    print("Loading candidate S1 pool from train_source1.tsv...")
    s1_pool = load_source_tsv(paths["train_source1"], "Train S1 Pool", "S1-", nrows=25000)
    
    # Assign match category
    categories = []
    for s1 in s1_pool["entity_id"]:
        mc = s1_match_counts.get(s1, 0)
        if mc == 0:
            cat = "singleton_0"
        elif mc == 1:
            cat = "single_1"
        elif mc <= 3:
            cat = "multi_2_3"
        else:
            cat = "multi_4_plus"
        categories.append(cat)

    s1_pool["match_category"] = categories
    s1_pool["strata"] = s1_pool["country"] + "__" + s1_pool["match_category"]

    print("Candidate strata distribution:")
    print(s1_pool["strata"].value_counts())

    # Sample exactly sample_size rows stratified by strata
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    sampled_idx, _ = next(sss.split(s1_pool, s1_pool["strata"]))
    sampled_s1 = s1_pool.iloc[sampled_idx].copy().reset_index(drop=True)

    print(f"\nSampled {len(sampled_s1)} S1 queries with distribution:")
    print(sampled_s1["strata"].value_counts())

    # Now split sampled_s1 into Train (60%), Tune (20%), Assessment (20%)
    # First: Train (60%) vs Remainder (40%)
    split1 = StratifiedShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
    train_idx, rem_idx = next(split1.split(sampled_s1, sampled_s1["strata"]))

    train_df = sampled_s1.iloc[train_idx].copy().reset_index(drop=True)
    rem_df = sampled_s1.iloc[rem_idx].copy().reset_index(drop=True)

    # Remainder split 50/50 into Tune and Assessment
    split2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    tune_idx, assess_idx = next(split2.split(rem_df, rem_df["strata"]))

    tune_df = rem_df.iloc[tune_idx].copy().reset_index(drop=True)
    assess_df = rem_df.iloc[assess_idx].copy().reset_index(drop=True)

    train_ids = train_df["entity_id"].tolist()
    tune_ids = tune_df["entity_id"].tolist()
    assess_ids = assess_df["entity_id"].tolist()

    # Integrity assertions: zero overlap
    assert len(set(train_ids).intersection(set(tune_ids))) == 0, "Train-Tune overlap detected!"
    assert len(set(train_ids).intersection(set(assess_ids))) == 0, "Train-Assess overlap detected!"
    assert len(set(tune_ids).intersection(set(assess_ids))) == 0, "Tune-Assess overlap detected!"
    assert len(train_ids) + len(tune_ids) + len(assess_ids) == sample_size, "Total count mismatch!"

    print(f"\nSplit Sizes:")
    print(f"  Train:      {len(train_ids):,} ({len(train_ids)/sample_size*100:.1f}%)")
    print(f"  Tune:       {len(tune_ids):,} ({len(tune_ids)/sample_size*100:.1f}%)")
    print(f"  Assessment: {len(assess_ids):,} ({len(assess_ids)/sample_size*100:.1f}%)")

    # Manifest
    manifest = {
        "sample_size": sample_size,
        "seed": seed,
        "train_size": len(train_ids),
        "tune_size": len(tune_ids),
        "assessment_size": len(assess_ids),
        "strata_summary": {
            "train": train_df["strata"].value_counts().to_dict(),
            "tune": tune_df["strata"].value_counts().to_dict(),
            "assessment": assess_df["strata"].value_counts().to_dict(),
        },
        "train_ids": train_ids,
        "tune_ids": tune_ids,
        "assessment_ids": assess_ids,
    }

    manifest_file = output_dir / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_file}")

    # Save dataframe as TSV
    sampled_s1.to_csv(output_dir / "sampled_s1.tsv", sep="\t", index=False)
    print("Saved sampled S1 data to sampled_s1.tsv")

    return manifest


if __name__ == "__main__":
    create_splits()
