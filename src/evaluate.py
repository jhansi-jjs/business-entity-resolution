"""Evaluation module for Business Entity Resolution Challenge.

Calculates exact entity-level Precision, Recall, and macro F0.5,
faithfully implementing singleton handling, unrounded internal values,
and comprehensive error and subset diagnostics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd


def compute_entity_metrics(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """Compute (precision, recall, f0_5) for a single Source 1 entity set match.

    Rules:
    - If true set is empty (singleton):
        - If pred set is empty: P=1.0, R=1.0, F0.5=1.0 (correct singleton).
        - If pred set is non-empty: P=0.0, R=0.0, F0.5=0.0 (false positive on singleton).
    - If true set is non-empty:
        - If pred set is empty: P=0.0, R=0.0, F0.5=0.0 (missed matches).
        - If pred set is non-empty:
            TP = len(true_set & pred_set)
            P = TP / len(pred_set)
            R = TP / len(true_set)
            denom = 0.25 * P + R
            F0.5 = (1.25 * P * R) / denom if denom > 0 else 0.0
    """
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0, 0.0, 0.0

    p = tp / len(pred_set)
    r = tp / len(true_set)
    denom = 0.25 * p + r
    f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

    return p, r, f05


def evaluate_predictions(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, Any]:
    """Calculate macro-averaged metrics across all evaluated Source 1 entities.

    Retains unrounded raw float values for precision thresholding and optimization,
    providing rounded keys for display.
    """
    p_list = []
    r_list = []
    f05_list = []

    singleton_total = 0
    singleton_correct = 0
    pred_match_counts = []

    for s1 in s1_ids:
        t_set = ground_truth_map.get(s1, set())
        p_set = predictions_map.get(s1, set())

        p, r, f05 = compute_entity_metrics(t_set, p_set)
        p_list.append(p)
        r_list.append(r)
        f05_list.append(f05)

        pred_match_counts.append(len(p_set))
        if not t_set:
            singleton_total += 1
            if not p_set:
                singleton_correct += 1

    p_arr = np.array(p_list, dtype=np.float64)
    r_arr = np.array(r_list, dtype=np.float64)
    f_arr = np.array(f05_list, dtype=np.float64)
    c_arr = np.array(pred_match_counts, dtype=np.int32)

    macro_p_raw = float(p_arr.mean()) if len(p_arr) > 0 else 0.0
    macro_r_raw = float(r_arr.mean()) if len(r_arr) > 0 else 0.0
    macro_f05_raw = float(f_arr.mean()) if len(f_arr) > 0 else 0.0

    # Report None (N/A) when no singletons are present
    if singleton_total > 0:
        singleton_acc_raw = float(singleton_correct / singleton_total)
        singleton_acc_disp = round(singleton_acc_raw, 4)
    else:
        singleton_acc_raw = None
        singleton_acc_disp = "N/A"

    return {
        # Full precision raw values for tuning / selection
        "macro_precision_raw": macro_p_raw,
        "macro_recall_raw": macro_r_raw,
        "macro_f0_5_raw": macro_f05_raw,
        "singleton_accuracy_raw": singleton_acc_raw,

        # Display friendly values
        "macro_precision": round(macro_p_raw, 4),
        "macro_recall": round(macro_r_raw, 4),
        "macro_f0_5": round(macro_f05_raw, 4),
        "singleton_count": singleton_total,
        "singleton_accuracy": singleton_acc_disp,
        "avg_predicted_matches": round(float(c_arr.mean()), 2) if len(c_arr) else 0.0,
        "pct_zero_predicted": round(float((c_arr == 0).mean() * 100), 2) if len(c_arr) else 0.0,
        "total_evaluated_s1": len(s1_ids)
    }


def evaluate_predictions_detailed(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    s1_metadata_df: pd.DataFrame
) -> Dict[str, Any]:
    """Detailed evaluation breaking down performance by country, match category, and missingness."""
    s1_ids = s1_metadata_df["entity_id"].tolist()
    overall = evaluate_predictions(predictions_map, ground_truth_map, s1_ids)

    breakdown = {"overall": overall, "by_country": {}, "by_match_cardinality": {}}

    # Breakdown by country
    for country, sub_df in s1_metadata_df.groupby("country"):
        sub_ids = sub_df["entity_id"].tolist()
        breakdown["by_country"][str(country)] = evaluate_predictions(predictions_map, ground_truth_map, sub_ids)

    # Breakdown by match count in GT
    s1_metadata_df = s1_metadata_df.copy()
    s1_metadata_df["true_match_count"] = [len(ground_truth_map.get(eid, set())) for eid in s1_ids]

    bins = [
        ("0 (Singleton)", s1_metadata_df[s1_metadata_df["true_match_count"] == 0]),
        ("1 match", s1_metadata_df[s1_metadata_df["true_match_count"] == 1]),
        ("2-3 matches", s1_metadata_df[s1_metadata_df["true_match_count"].isin([2, 3])]),
        ("4+ matches", s1_metadata_df[s1_metadata_df["true_match_count"] >= 4]),
    ]

    for label, sub_df in bins:
        if len(sub_df) > 0:
            breakdown["by_match_cardinality"][label] = evaluate_predictions(
                predictions_map, ground_truth_map, sub_df["entity_id"].tolist()
            )

    return breakdown


def compute_heuristic_score(feat_dict: Dict[str, float]) -> float:
    """Interpretable heuristic baseline combination."""
    name_score = (
        0.30 * feat_dict.get("feat_name_fuzz_ratio", 0.0) +
        0.30 * feat_dict.get("feat_name_core_exact", 0.0) +
        0.20 * feat_dict.get("feat_name_token_set_ratio", 0.0) +
        0.20 * feat_dict.get("feat_name_containment", 0.0)
    )
    addr_score = (
        0.40 * feat_dict.get("feat_addr_fuzz_ratio", 0.0) +
        0.30 * feat_dict.get("feat_addr_token_set_ratio", 0.0) +
        0.30 * feat_dict.get("feat_addr_numeric_jaccard", 0.0)
    )
    if feat_dict.get("feat_addr_is_empty", 0.0) > 0.5:
        return name_score * 0.90
    else:
        return 0.65 * name_score + 0.35 * addr_score
