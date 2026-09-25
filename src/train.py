"""Model training, grouped validation, and threshold tuning module.

Trains candidate matchers with GroupShuffleSplit on source1_entity_id,
evaluates entity-level macro F0.5, tunes the decision threshold, and saves artifacts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import lightgbm as lgb

from src.features import FEATURE_NAMES, compute_pair_features, enrich_record_dict
from src.evaluate import evaluate_predictions, compute_heuristic_score


def build_pair_dataset(
    s1_dict: Dict[str, Dict[str, Any]],
    cand_dict: Dict[str, Dict[str, Any]],
    candidate_tuples: List[Tuple],
    gt_map: Dict[str, Set[str]]
) -> pd.DataFrame:
    """Construct labeled feature dataframe from candidate pairs."""
    records = []
    for item in candidate_tuples:
        s1_id = item[0]
        cid = item[1]
        csrc = item[2]
        name_score = item[3]
        addr_score = item[4]
        num_methods = item[6]

        s1_row = s1_dict[s1_id]
        cand_row = cand_dict[cid]

        feats = compute_pair_features(
            s1_row, cand_row,
            name_retrieval_score=name_score,
            addr_retrieval_score=addr_score,
            num_blocking_methods=num_methods
        )

        true_set = gt_map.get(s1_id, set())
        label = 1 if cid in true_set else 0

        row = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cid,
            "label": label,
        }
        row.update(feats)
        records.append(row)

    return pd.DataFrame(records)


def tune_threshold_and_evaluate(
    model: Any,
    val_df: pd.DataFrame,
    val_s1_ids: List[str],
    gt_map: Dict[str, Set[str]],
    is_heuristic: bool = False
) -> Tuple[float, Dict[str, float]]:
    """Grid search thresholds for max macro F0.5 at entity level."""
    # Predict probabilities for all pairs
    if is_heuristic:
        probs = [compute_heuristic_score(row) for _, row in val_df.iterrows()]
    else:
        probs = model.predict_proba(val_df[FEATURE_NAMES])[:, 1]

    val_df = val_df.copy()
    val_df["pred_prob"] = probs

    best_thresh = 0.5
    best_metrics = {}
    best_f05 = -1.0

    thresholds = np.linspace(0.15, 0.90, 16)
    for t in thresholds:
        t = round(float(t), 2)
        # Group predictions by s1_id
        pred_map = {}
        filtered = val_df[val_df["pred_prob"] >= t]
        for s1, group in filtered.groupby("source1_entity_id"):
            pred_map[s1] = set(group["candidate_entity_id"])

        metrics = evaluate_predictions(pred_map, gt_map, val_s1_ids)
        if metrics["macro_f0_5"] > best_f05:
            best_f05 = metrics["macro_f0_5"]
            best_thresh = t
            best_metrics = metrics

    return best_thresh, best_metrics


def train_and_compare_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_s1_ids: List[str],
    gt_map: Dict[str, Set[str]],
    output_dir: Path
) -> pd.DataFrame:
    """Train candidates, compare grouped macro F0.5, and save the best model."""
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"]

    print(f"Training pairs: {len(X_train):,} (Positives: {y_train.sum():,}, Negatives: {(len(y_train) - y_train.sum()):,})")
    print(f"Validation pairs: {len(val_df):,}")

    models_to_test = {
        "Heuristic Baseline": None,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1),
    }

    results = []
    best_model_name = ""
    best_model_obj = None
    best_f05 = -1.0
    best_threshold = 0.5
    best_final_metrics = {}

    for name, clf in models_to_test.items():
        print(f"\n--- Evaluating {name} ---")
        if clf is not None:
            clf.fit(X_train, y_train)

        thresh, metrics = tune_threshold_and_evaluate(
            clf, val_df, val_s1_ids, gt_map, is_heuristic=(clf is None)
        )

        print(f"  Best Threshold: {thresh}")
        print(f"  Macro Precision: {metrics['macro_precision']}")
        print(f"  Macro Recall:    {metrics['macro_recall']}")
        print(f"  Macro F0.5:      {metrics['macro_f0_5']}")
        print(f"  Singleton Acc:   {metrics['singleton_accuracy']}")

        res_entry = {
            "model": name,
            "threshold": thresh,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f0_5": metrics["macro_f0_5"],
            "singleton_accuracy": metrics["singleton_accuracy"],
            "avg_predicted_matches": metrics["avg_predicted_matches"]
        }
        results.append(res_entry)

        if metrics["macro_f0_5"] > best_f05:
            best_f05 = metrics["macro_f0_5"]
            best_model_name = name
            best_model_obj = clf
            best_threshold = thresh
            best_final_metrics = metrics

    # Save experiments CSV
    exp_df = pd.DataFrame(results)
    exp_path = PROJECT_ROOT / "experiments" / "experiments.csv"
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    exp_df.to_csv(exp_path, index=False)
    print(f"\nExperiments log saved to {exp_path}")

    # Save best model and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_model_obj is not None:
        model_path = output_dir / "matcher.joblib"
        joblib.dump(best_model_obj, model_path)
        print(f"Best model ({best_model_name}) saved to {model_path}")

    config = {
        "best_model": best_model_name,
        "threshold": best_threshold,
        "feature_names": FEATURE_NAMES,
        "metrics": best_final_metrics
    }
    config_path = output_dir / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Model configuration saved to {config_path}")

    return exp_df
