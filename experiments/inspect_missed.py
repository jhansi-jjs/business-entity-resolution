"""Inspect missed matches and false positives on the held-out assessment set."""

import sys
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import get_default_paths, load_ground_truth
from src.features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    benchmark_pairs_path = PROJECT_ROOT / "experiments" / "splits" / "benchmark_pairs.tsv"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"
    model_path = PROJECT_ROOT / "models" / "matcher.joblib"
    config_path = PROJECT_ROOT / "models" / "model_config.json"

    if not manifest_path.exists() or not benchmark_pairs_path.exists():
        raise FileNotFoundError("Prerequisite experiment files not found. Run run_training_experiments.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    clf = joblib.load(model_path)
    threshold = float(config["threshold"])
    assessment_ids = set(manifest["assessment_ids"])

    # Load S1 queries
    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    s1_dict = {row["entity_id"]: row.to_dict() for _, row in s1_df.iterrows()}

    # Load Ground Truth
    paths = get_default_paths()
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(assessment_ids)]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}
    for s1 in assessment_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()

    # Load Pairs for assessment
    pairs_df = pd.read_csv(benchmark_pairs_path, sep="\t")
    assess_pairs = pairs_df[pairs_df["source1_entity_id"].isin(assessment_ids)].copy().reset_index(drop=True)

    probs = clf.predict_proba(assess_pairs[FEATURE_NAMES])[:, 1]
    assess_pairs["pred_prob"] = probs
    assess_pairs["predicted"] = (probs >= threshold).astype(int)

    # 1. Retrieved candidates by query
    retrieved_by_s1 = {}
    for s1, grp in assess_pairs.groupby("source1_entity_id"):
        retrieved_by_s1[s1] = set(grp["candidate_entity_id"])

    # 2. Predicted matches by query
    predicted_by_s1 = {}
    for s1 in assessment_ids:
        predicted_by_s1[s1] = set()
    for s1, grp in assess_pairs[assess_pairs["predicted"] == 1].groupby("source1_entity_id"):
        predicted_by_s1[s1] = set(grp["candidate_entity_id"])

    # Analysis categories
    blocking_misses = []
    classifier_false_negatives = []
    classifier_false_positives = []
    singleton_false_positives = []

    for s1 in assessment_ids:
        true_set = gt_map.get(s1, set())
        pred_set = predicted_by_s1.get(s1, set())
        retrieved_set = retrieved_by_s1.get(s1, set())
        s1_info = s1_dict.get(s1, {})

        # Singleton false positives
        if len(true_set) == 0 and len(pred_set) > 0:
            singleton_false_positives.append({
                "source1_id": s1,
                "s1_name": s1_info.get("business_name", ""),
                "s1_addr": s1_info.get("business_address", ""),
                "country": s1_info.get("country", ""),
                "false_matches": list(pred_set)
            })

        # Blocking misses: in true_set, but not in retrieved_set
        for tid in true_set:
            if tid not in retrieved_set:
                blocking_misses.append({
                    "source1_id": s1,
                    "target_id": tid,
                    "s1_name": s1_info.get("business_name", ""),
                    "s1_addr": s1_info.get("business_address", ""),
                    "country": s1_info.get("country", "")
                })

        # Classifier false negatives: in retrieved_set, in true_set, but predicted = 0
        for tid in true_set:
            if tid in retrieved_set and tid not in pred_set:
                pair_row = assess_pairs[(assess_pairs["source1_entity_id"] == s1) & (assess_pairs["candidate_entity_id"] == tid)]
                prob = float(pair_row["pred_prob"].iloc[0]) if len(pair_row) > 0 else 0.0
                classifier_false_negatives.append({
                    "source1_id": s1,
                    "target_id": tid,
                    "prob": prob,
                    "threshold": threshold,
                    "s1_name": s1_info.get("business_name", ""),
                    "s1_addr": s1_info.get("business_address", ""),
                    "country": s1_info.get("country", "")
                })

        # Classifier false positives: in pred_set, but not in true_set
        for pid in pred_set:
            if pid not in true_set:
                pair_row = assess_pairs[(assess_pairs["source1_entity_id"] == s1) & (assess_pairs["candidate_entity_id"] == pid)]
                prob = float(pair_row["pred_prob"].iloc[0]) if len(pair_row) > 0 else 0.0
                classifier_false_positives.append({
                    "source1_id": s1,
                    "target_id": pid,
                    "prob": prob,
                    "s1_name": s1_info.get("business_name", ""),
                    "s1_addr": s1_info.get("business_address", ""),
                    "country": s1_info.get("country", "")
                })

    print("\n" + "="*70)
    print("HELD-OUT ASSESSMENT ERROR ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total Assessment Queries:              {len(assessment_ids):,}")
    print(f"Blocking Misses (Recall Bottleneck):   {len(blocking_misses):,}")
    print(f"Classifier False Negatives:            {len(classifier_false_negatives):,}")
    print(f"Classifier False Positives:            {len(classifier_false_positives):,}")
    print(f"Singleton Queries with False Matches:  {len(singleton_false_positives):,}")

    error_report = {
        "num_assessment_queries": len(assessment_ids),
        "total_blocking_misses": len(blocking_misses),
        "total_classifier_fn": len(classifier_false_negatives),
        "total_classifier_fp": len(classifier_false_positives),
        "singleton_fps": len(singleton_false_positives),
        "sample_blocking_misses": blocking_misses[:5],
        "sample_classifier_fn": classifier_false_negatives[:5],
        "sample_classifier_fp": classifier_false_positives[:5],
        "sample_singleton_fp": singleton_false_positives[:5],
    }

    report_path = PROJECT_ROOT / "experiments" / "error_analysis.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(error_report, f, indent=2)
    print(f"\nSaved detailed error analysis to {report_path}")


if __name__ == "__main__":
    main()
