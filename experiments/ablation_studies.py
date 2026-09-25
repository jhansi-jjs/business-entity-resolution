"""Real feature ablation studies with model retraining and grouped tuning."""

import sys
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.features import FEATURE_NAMES
from src.train import tune_threshold_and_evaluate
from src.data_loader import get_default_paths, load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    benchmark_pairs_path = PROJECT_ROOT / "experiments" / "splits" / "benchmark_pairs.tsv"

    if not manifest_path.exists() or not benchmark_pairs_path.exists():
        raise FileNotFoundError(
            "Benchmark files not found! Run experiments/run_training_experiments.py first."
        )

    print("\n" + "="*70)
    print("STARTING REAL FEATURE ABLATION STUDIES")
    print("="*70)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ids = set(manifest["train_ids"])
    tune_ids = set(manifest["tune_ids"])
    tune_s1_list = list(tune_ids)

    print(f"Loading benchmark pairs from {benchmark_pairs_path}...")
    pair_df = pd.read_csv(benchmark_pairs_path, sep="\t")
    print(f"Loaded {len(pair_df):,} total candidate pairs.")

    train_df = pair_df[pair_df["source1_entity_id"].isin(train_ids)].copy().reset_index(drop=True)
    tune_df = pair_df[pair_df["source1_entity_id"].isin(tune_ids)].copy().reset_index(drop=True)

    print(f"Train pairs: {len(train_df):,}, Tune pairs: {len(tune_df):,}")

    # Build GT map for tune set
    paths = get_default_paths()
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(tune_ids)]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}
    for s1 in tune_s1_list:
        if s1 not in gt_map:
            gt_map[s1] = set()

    # Define feature groups
    name_feats = [f for f in FEATURE_NAMES if f.startswith("feat_name_")]
    addr_feats = [f for f in FEATURE_NAMES if f.startswith("feat_addr_")]
    name_addr_feats = name_feats + addr_feats
    no_meta_feats = [f for f in FEATURE_NAMES if not f.startswith("feat_num_") and "retrieval" not in f]
    full_feats = FEATURE_NAMES

    ablation_groups = {
        "A. Name features only": name_feats,
        "B. Address features only": addr_feats,
        "C. Name + Address basic": name_addr_feats,
        "D. Name + Address + Numeric (no meta)": no_meta_feats,
        "E. Full Features (All Signals)": full_feats,
    }

    results = []
    y_train = train_df["label"]

    for name, feat_subset in ablation_groups.items():
        print(f"\n--- Running Ablation: {name} ({len(feat_subset)} features) ---")
        X_train = train_df[feat_subset]

        # Train model on this subset
        clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        # Tune threshold and evaluate on Tune split
        thresh, metrics = tune_threshold_and_evaluate(
            model=clf,
            val_df=tune_df,
            val_s1_ids=tune_s1_list,
            gt_map=gt_map,
            feature_names=feat_subset
        )

        print(f"  Best Threshold: {thresh}")
        print(f"  Macro Precision: {metrics['macro_precision']}")
        print(f"  Macro Recall:    {metrics['macro_recall']}")
        print(f"  Macro F0.5:      {metrics['macro_f0_5']}")
        print(f"  Singleton Acc:   {metrics['singleton_accuracy']}")

        results.append({
            "ablation": name,
            "num_features": len(feat_subset),
            "best_threshold": thresh,
            "val_macro_precision": metrics["macro_precision"],
            "val_macro_recall": metrics["macro_recall"],
            "val_macro_f0_5": metrics["macro_f0_5"],
            "val_singleton_accuracy": metrics["singleton_accuracy"],
        })

    abl_df = pd.DataFrame(results)
    out_file = PROJECT_ROOT / "experiments" / "ablation_results.csv"
    abl_df.to_csv(out_file, index=False)
    print("\n" + "="*70)
    print("GENUINE MEASURED FEATURE ABLATION RESULTS:")
    print("="*70)
    print(abl_df.to_string(index=False))
    print("="*70)
    print(f"Saved real ablation results to {out_file}")


if __name__ == "__main__":
    main()
