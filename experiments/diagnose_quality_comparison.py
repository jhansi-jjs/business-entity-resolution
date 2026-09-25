"""Quality Comparison and Error Diagnosis: Deterministic Baseline vs GBDT Artifact.

Evaluates both approaches on the reproducible, representative Tune split (500 queries)
against organizer ground truth to determine exact sources of error.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_predictions, compute_entity_metrics

# Load split manifest
with open("experiments/splits/split_manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)
tune_ids = set(manifest["tune_ids"])

# Load sampled S1 metadata
s1_meta = pd.read_csv("experiments/splits/sampled_s1.tsv", sep="\t")
tune_s1 = s1_meta[s1_meta["entity_id"].isin(tune_ids)].set_index("entity_id")

# Load Ground Truth
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
tune_gt_df = gt_df[gt_df["source1_entity_id"].isin(tune_ids)]
ground_truth = {
    row["source1_entity_id"]: set(str(row["matched_entity_ids"]).split(","))
    if pd.notna(row["matched_entity_ids"]) and str(row["matched_entity_ids"]).strip()
    else set()
    for _, row in tune_gt_df.iterrows()
}
for eid in tune_ids:
    if eid not in ground_truth:
        ground_truth[eid] = set()

# Load benchmark candidate pairs for tune_ids
pairs_df = pd.read_csv("experiments/splits/benchmark_pairs.tsv", sep="\t")
tune_pairs = pairs_df[pairs_df["source1_entity_id"].isin(tune_ids)].copy()

# Load trained GBDT model
model = joblib.load("models/matcher.joblib")
with open("models/model_config.json", "r") as f:
    cfg = json.load(f)
feature_names = cfg["feature_names"]
threshold = float(cfg.get("threshold", 0.75))

# 1. GBDT Predictions
probs = model.predict_proba(tune_pairs[feature_names])[:, 1]
tune_pairs["gbdt_prob"] = probs

gbdt_preds = {eid: set() for eid in tune_ids}
for _, row in tune_pairs[tune_pairs["gbdt_prob"] >= threshold].iterrows():
    gbdt_preds[row["source1_entity_id"]].add(row["candidate_entity_id"])

# 2. Deterministic Baseline Predictions on the same pairs
det_preds = {eid: set() for eid in tune_ids}
for _, row in tune_pairs.iterrows():
    is_match = False
    if row.get("feat_addr_postal_match", 0.0) == 1.0:
        is_match = True
    elif row.get("feat_addr_numeric_shared", 0.0) >= 1.0 and row.get("feat_addr_numeric_jaccard", 0.0) >= 0.5:
        is_match = True
    elif row.get("feat_name_exact", 0.0) == 1.0:
        is_match = True
    elif row.get("feat_name_core_exact", 0.0) == 1.0 and row.get("feat_name_len_diff", 0.0) <= 2:
        is_match = True

    if is_match:
        det_preds[row["source1_entity_id"]].add(row["candidate_entity_id"])

# Evaluate both
tune_ids_list = list(tune_ids)
gbdt_eval = evaluate_predictions(gbdt_preds, ground_truth, tune_ids_list)
det_eval = evaluate_predictions(det_preds, ground_truth, tune_ids_list)

print("=" * 80)
print("TUNE SPLIT EVALUATION (500 QUERIES) COMPARISON")
print("=" * 80)
print(f"GBDT Macro F0.5:        {gbdt_eval['macro_f0_5']:.4f}")
print(f"GBDT Macro Precision:   {gbdt_eval['macro_precision']:.4f}")
print(f"GBDT Macro Recall:      {gbdt_eval['macro_recall']:.4f}")
print(f"GBDT Singleton Acc:     {gbdt_eval['singleton_accuracy']:.4f}")
print(f"GBDT Avg Match Count:   {gbdt_eval['avg_predicted_matches']:.2f}")
print("-" * 40)
print(f"Baseline Macro F0.5:    {det_eval['macro_f0_5']:.4f}")
print(f"Baseline Macro Prec:    {det_eval['macro_precision']:.4f}")
print(f"Baseline Macro Recall:  {det_eval['macro_recall']:.4f}")
print(f"Baseline Singleton Acc: {det_eval['singleton_accuracy']:.4f}")
print(f"Baseline Avg Matches:   {det_eval['avg_predicted_matches']:.2f}")

# Breakdown by Country
print("\n" + "=" * 80)
print("BREAKDOWN BY COUNTRY")
print("=" * 80)
for country in ["US", "India"]:
    c_eids = list(tune_s1[tune_s1["country"] == country].index)
    c_gt = {eid: ground_truth[eid] for eid in c_eids}
    c_gbdt = {eid: gbdt_preds[eid] for eid in c_eids}
    c_det = {eid: det_preds[eid] for eid in c_eids}

    g_res = evaluate_predictions(c_gbdt, c_gt, c_eids)
    d_res = evaluate_predictions(c_det, c_gt, c_eids)
    print(f"Country: {country} ({len(c_eids)} queries)")
    print(f"  GBDT     -> F0.5: {g_res['macro_f0_5']:.4f}, Prec: {g_res['macro_precision']:.4f}, Rec: {g_res['macro_recall']:.4f}, Avg Matches: {g_res['avg_predicted_matches']:.2f}")
    print(f"  Baseline -> F0.5: {d_res['macro_f0_5']:.4f}, Prec: {d_res['macro_precision']:.4f}, Rec: {d_res['macro_recall']:.4f}, Avg Matches: {d_res['avg_predicted_matches']:.2f}")

# Breakdown by Match Group
print("\n" + "=" * 80)
print("BREAKDOWN BY MATCH GROUP")
print("=" * 80)
groups = {
    "Singleton (0 matches)": [eid for eid in tune_ids if len(ground_truth[eid]) == 0],
    "Single Match (1)": [eid for eid in tune_ids if len(ground_truth[eid]) == 1],
    "Multi Match (2-3)": [eid for eid in tune_ids if 2 <= len(ground_truth[eid]) <= 3],
    "Multi Match (4+)": [eid for eid in tune_ids if len(ground_truth[eid]) >= 4],
}
for grp_name, eids in groups.items():
    grp_gt = {eid: ground_truth[eid] for eid in eids}
    grp_gbdt = {eid: gbdt_preds[eid] for eid in eids}
    grp_det = {eid: det_preds[eid] for eid in eids}

    g_res = evaluate_predictions(grp_gbdt, grp_gt, eids)
    d_res = evaluate_predictions(grp_det, grp_gt, eids)
    print(f"Group: {grp_name} ({len(eids)} queries)")
    print(f"  GBDT     -> F0.5: {g_res['macro_f0_5']:.4f}, Prec: {g_res['macro_precision']:.4f}, Rec: {g_res['macro_recall']:.4f}, Avg Matches: {g_res['avg_predicted_matches']:.2f}")
    print(f"  Baseline -> F0.5: {d_res['macro_f0_5']:.4f}, Prec: {d_res['macro_precision']:.4f}, Rec: {d_res['macro_recall']:.4f}, Avg Matches: {d_res['avg_predicted_matches']:.2f}")

# Threshold Sensitivity Analysis on GBDT
print("\n" + "=" * 80)
print("GBDT THRESHOLD SENSITIVITY ON TUNE SPLIT")
print("=" * 80)
for thr in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    t_preds = {eid: set() for eid in tune_ids}
    for _, row in tune_pairs[tune_pairs["gbdt_prob"] >= thr].iterrows():
        t_preds[row["source1_entity_id"]].add(row["candidate_entity_id"])
    t_eval = evaluate_predictions(t_preds, ground_truth, tune_ids_list)
    print(f"  Threshold {thr:.2f} -> Macro F0.5: {t_eval['macro_f0_5']:.4f} | Prec: {t_eval['macro_precision']:.4f} | Rec: {t_eval['macro_recall']:.4f} | Singleton Acc: {t_eval['singleton_accuracy']:.4f} | Avg Matches: {t_eval['avg_predicted_matches']:.2f}")

