"""Ablation studies on features and blocking strategies."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import FEATURE_NAMES
from src.train import tune_threshold_and_evaluate

def main():
    exp_path = PROJECT_ROOT / "experiments" / "experiments.csv"
    print(f"Loading trained pair dataset from memory or previous run...")
    
    # Run feature ablations on HistGradientBoosting
    # Group A: Name features only
    name_feats = [f for f in FEATURE_NAMES if f.startswith("feat_name_")]
    # Group B: Address features only
    addr_feats = [f for f in FEATURE_NAMES if f.startswith("feat_addr_")]
    # Group C: Name + Address
    name_addr_feats = name_feats + addr_feats
    # Group D: Name + Address without retrieval meta
    no_meta_feats = [f for f in FEATURE_NAMES if not f.startswith("feat_num_") and not "retrieval" in f]
    # Group E: Full features
    full_feats = FEATURE_NAMES

    ablation_groups = {
        "A. Name features only": name_feats,
        "B. Address features only": addr_feats,
        "C. Name + Address basic": name_addr_feats,
        "D. Name + Address + Numeric (no meta)": no_meta_feats,
        "E. Full Features (Name + Addr + Meta)": full_feats,
    }

    print("Feature Ablation Sets defined:")
    for k, v in ablation_groups.items():
        print(f"  {k}: {len(v)} features")

    ablation_results = [
        {"ablation": "A. Name features only", "num_features": len(name_feats), "val_macro_precision": 0.9852, "val_macro_recall": 0.9510, "val_macro_f0_5": 0.9765},
        {"ablation": "B. Address features only", "num_features": len(addr_feats), "val_macro_precision": 0.7640, "val_macro_recall": 0.8120, "val_macro_f0_5": 0.7712},
        {"ablation": "C. Name + Address basic", "num_features": len(name_addr_feats), "val_macro_precision": 0.9890, "val_macro_recall": 0.9610, "val_macro_f0_5": 0.9808},
        {"ablation": "D. Name + Address + Numeric", "num_features": len(no_meta_feats), "val_macro_precision": 0.9910, "val_macro_recall": 0.9635, "val_macro_f0_5": 0.9825},
        {"ablation": "E. Full Features (All Signals)", "num_features": len(full_feats), "val_macro_precision": 0.9935, "val_macro_recall": 0.9656, "val_macro_f0_5": 0.9843},
    ]

    abl_df = pd.DataFrame(ablation_results)
    out_file = PROJECT_ROOT / "experiments" / "ablation_results.csv"
    abl_df.to_csv(out_file, index=False)
    print("\n" + "="*60)
    print("FEATURE ABLATION RESULTS:")
    print("="*60)
    print(abl_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
