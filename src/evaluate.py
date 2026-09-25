"""Evaluation module for Business Entity Resolution Challenge.

Calculates exact entity-level Precision, Recall, and macro F0.5,
faithfully implementing singleton handling and set-level metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd


def compute_entity_metrics(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """Compute (precision, recall, f0_5) for a single Source 1 entity set match."""
    # Singleton case: true set is empty
    if not true_set:
        if not pred_set:
            return 1.0, 1.0, 1.0  # Perfect credit for correctly predicting no match
        else:
            return 0.0, 0.0, 0.0  # False positive on a singleton

    # True set is non-empty, but prediction is empty
    if not pred_set:
        return 0.0, 0.0, 0.0

    # Both non-empty
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
) -> Dict[str, float]:
    """Calculate macro-averaged metrics across all evaluated Source 1 entities."""
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

    p_arr = np.array(p_list)
    r_arr = np.array(r_list)
    f_arr = np.array(f05_list)
    c_arr = np.array(pred_match_counts)

    return {
        "macro_precision": round(float(p_arr.mean()), 4),
        "macro_recall": round(float(r_arr.mean()), 4),
        "macro_f0_5": round(float(f_arr.mean()), 4),
        "singleton_count": singleton_total,
        "singleton_accuracy": round(singleton_correct / singleton_total, 4) if singleton_total > 0 else 1.0,
        "avg_predicted_matches": round(float(c_arr.mean()), 2),
        "pct_zero_predicted": round(float((c_arr == 0).mean() * 100), 2),
        "total_evaluated_s1": len(s1_ids)
    }


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
        # Address missing in candidate: rely strictly on high name confidence
        return name_score * 0.90
    else:
        return 0.65 * name_score + 0.35 * addr_score


if __name__ == "__main__":
    # Test evaluation logic
    dummy_gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set(),  # singleton
        "S1-4": set(),  # singleton
    }
    dummy_pred = {
        "S1-1": {"S2-10", "S3-20"},  # Perfect: P=1, R=1, F0.5=1
        "S1-2": {"S2-30", "S3-99"},  # 1 hit, 1 false positive: P=0.5, R=1, F0.5=0.555
        "S1-3": set(),               # Correct singleton: 1.0
        "S1-4": {"S2-50"},           # False positive singleton: 0.0
    }
    res = evaluate_predictions(dummy_pred, dummy_gt, ["S1-1", "S1-2", "S1-3", "S1-4"])
    print("=== Evaluator Sanity Verification ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
