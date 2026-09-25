"""Test fast inverted-index blocker recall and throughput on Tune queries."""

import sys
from pathlib import Path
import json
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_name, normalize_address
from src.data_loader import get_default_paths, load_ground_truth

def main():
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    benchmark_pairs_path = PROJECT_ROOT / "experiments" / "splits" / "benchmark_pairs.tsv"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tune_ids = set(manifest["tune_ids"])
    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    tune_s1 = s1_df[s1_df["entity_id"].isin(tune_ids)].copy().reset_index(drop=True)

    paths = get_default_paths()
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(tune_ids)]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}

    print(f"Tune queries: {len(tune_s1):,}, Total true links: {sum(len(v) for v in gt_map.values()):,}")

    # Check ground truth target names vs S1 names
    # Load targets from benchmark_pairs
    pair_df = pd.read_csv(benchmark_pairs_path, sep="\t")
    tune_pairs = pair_df[pair_df["source1_entity_id"].isin(tune_ids)]
    true_pairs = tune_pairs[tune_pairs["label"] == 1]
    print(f"True pairs in benchmark dataset: {len(true_pairs):,}")

    # Inspect name similarity distribution on true pairs
    print("Exact name core matches among true pairs:", (true_pairs["feat_name_core_exact"] == 1).mean())
    print("Fuzz ratio >= 80 among true pairs:", (true_pairs["feat_name_fuzz_ratio"] >= 0.80).mean())
    print("Token sort ratio >= 80 among true pairs:", (true_pairs["feat_name_token_sort_ratio"] >= 0.80).mean())
    print("Token containment >= 80 among true pairs:", (true_pairs["feat_name_containment"] >= 0.80).mean())

if __name__ == "__main__":
    main()
