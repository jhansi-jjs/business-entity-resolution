"""Bounded Experiment: Train and Evaluate with Realistic Full-Target Negatives.

Retrieves candidates for training queries against the FULL uncurated training target index
WITHOUT ground-truth injection or target curation. Ground truth is used solely to label
retrieved pairs. Trains a separate model (models/matcher_hard_negatives.joblib) and evaluates
both models on identical held-out tuning queries using full-index retrieval.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_default_paths, iter_source_tsv, load_ground_truth
from src.normalize import normalize_name, normalize_address, extract_postal_code
from src.blocking import CountryBlocker
from src.features import compute_pair_features, enrich_record_dict, FEATURE_NAMES
from src.evaluate import evaluate_predictions, compute_entity_metrics


def run_hard_negatives_experiment():
    print("=" * 80)
    print("EXPERIMENT: FULL-INDEX RETRIEVAL TRAINING WITH REALISTIC HARD NEGATIVES")
    print("=" * 80)

    paths = get_default_paths()
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ids_all = set(manifest["train_ids"])
    tune_ids_all = set(manifest["tune_ids"])

    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    gt_df = load_ground_truth(paths["train_gt"])

    # 1. Select representative India training (100) and tuning (50) queries
    # Strictly preserving query isolation (zero overlap with assessment_ids)
    india_train = s1_df[(s1_df["entity_id"].isin(train_ids_all)) & (s1_df["country"] == "India")].head(100)
    india_tune = s1_df[(s1_df["entity_id"].isin(tune_ids_all)) & (s1_df["country"] == "India")].head(50)

    print(f"Selected {len(india_train)} India training queries and {len(india_tune)} India tuning queries.")

    # Build Ground Truth Map
    all_q_ids = set(india_train["entity_id"]).union(set(india_tune["entity_id"]))
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(all_q_ids)]
    gt_map: Dict[str, Set[str]] = {eid: set() for eid in all_q_ids}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}

    # 2. Build full India target index (4.13M uncurated records from S2 and S3)
    # ZERO ground truth injection - 100% realistic search space
    print("\n[1/4] Building full uncurated India target index from Source 2 and Source 3...")
    t0 = time.time()
    s2_list = []
    for c in iter_source_tsv(paths["train_source2"], chunksize=250000):
        hit = c[c["country"] == "India"][["entity_id", "business_name", "business_address", "country"]]
        if len(hit) > 0:
            s2_list.append(hit)

    s3_list = []
    for c in iter_source_tsv(paths["train_source3"], chunksize=250000):
        hit = c[c["country"] == "India"][["entity_id", "business_name", "business_address", "country"]]
        if len(hit) > 0:
            s3_list.append(hit)

    targets_india = pd.concat(s2_list + s3_list, ignore_index=True).drop_duplicates(subset=["entity_id"])
    del s2_list, s3_list
    gc.collect()

    print(f"  Loaded {len(targets_india):,} uncurated India targets in {time.time()-t0:.1f}s.")

    # Fit inverted blocker on targets
    targets_india["name_core"] = [normalize_name(n)[1] for n in targets_india["business_name"]]
    targets_india["addr_norm"] = [normalize_address(a)[0] for a in targets_india["business_address"]]

    blocker = CountryBlocker(
        country="India",
        name_top_k=25,
        addr_top_k=8,
        name_sim_threshold=0.10,
        addr_sim_threshold=0.18,
        backend="inverted"
    )
    t_fit0 = time.time()
    blocker.fit(targets_india)
    print(f"  Blocker index built in {time.time()-t_fit0:.1f}s.")

    target_lookup = {
        eid: (n, a, c) for eid, n, a, c in zip(
            targets_india["entity_id"].values,
            targets_india["business_name"].fillna("").values,
            targets_india["business_address"].fillna("").values,
            targets_india["country"].fillna("").values
        )
    }

    # 3. Retrieve realistic candidate pairs for training queries
    print("\n[2/4] Retrieving realistic candidate pairs for training queries...")
    india_train["name_core"] = [normalize_name(n)[1] for n in india_train["business_name"]]
    india_train["addr_norm"] = [normalize_address(a)[0] for a in india_train["business_address"]]

    train_cands = blocker.query(india_train, batch_size=50)
    print(f"  Retrieved {len(train_cands):,} candidate pairs across {len(india_train)} training queries.")

    # Featurize training pairs
    train_records = []
    train_labels = []
    train_s1_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in india_train.iterrows()}

    needed_tids = {item[1] for item in train_cands if item[1] in target_lookup}
    target_dict = {
        tid: enrich_record_dict({
            "entity_id": tid,
            "business_name": target_lookup[tid][0],
            "business_address": target_lookup[tid][1],
            "country": target_lookup[tid][2]
        }) for tid in needed_tids
    }

    for item in train_cands:
        s1_id, cid, csrc, n_score, a_score, _, num_m = item
        if cid not in target_dict or s1_id not in train_s1_dict:
            continue
        feats = compute_pair_features(
            train_s1_dict[s1_id],
            target_dict[cid],
            name_retrieval_score=n_score,
            addr_retrieval_score=a_score,
            num_blocking_methods=num_m
        )
        train_records.append(feats)
        is_true = 1 if cid in gt_map.get(s1_id, set()) else 0
        train_labels.append(is_true)

    train_X = pd.DataFrame(train_records)[FEATURE_NAMES]
    train_y = np.array(train_labels)
    n_pos = int(train_y.sum())
    n_neg = len(train_y) - n_pos
    print(f"  Realistic Training Set: {len(train_X):,} pairs (Pos: {n_pos:,}, Neg: {n_neg:,}, Pos Ratio: {n_pos/len(train_X):.4f})")

    # 4. Train New GBDT Model with Hard Negatives
    print("\n[3/4] Training new HistGradientBoostingClassifier on realistic full-target negatives...")
    new_model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
    new_model.fit(train_X, train_y)
    models_dir = PROJECT_ROOT / "models"
    new_model_path = models_dir / "matcher_hard_negatives.joblib"
    joblib.dump(new_model, new_model_path)
    print(f"  Model saved to {new_model_path}")

    # 5. Evaluate Both Models on Identical Held-Out Tuning Queries
    print("\n[4/4] Evaluating existing vs new model on identical 50 tuning queries against full index...")
    india_tune["name_core"] = [normalize_name(n)[1] for n in india_tune["business_name"]]
    india_tune["addr_norm"] = [normalize_address(a)[0] for a in india_tune["business_address"]]

    tune_cands = blocker.query(india_tune, batch_size=50)
    print(f"  Retrieved {len(tune_cands):,} candidate pairs for 50 tuning queries.")

    tune_s1_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in india_tune.iterrows()}
    needed_tune_tids = {item[1] for item in tune_cands if item[1] in target_lookup}
    for tid in needed_tune_tids:
        if tid not in target_dict:
            target_dict[tid] = enrich_record_dict({
                "entity_id": tid,
                "business_name": target_lookup[tid][0],
                "business_address": target_lookup[tid][1],
                "country": target_lookup[tid][2]
            })

    tune_records = []
    tune_meta = []
    for item in tune_cands:
        s1_id, cid, csrc, n_score, a_score, _, num_m = item
        if cid not in target_dict or s1_id not in tune_s1_dict:
            continue
        feats = compute_pair_features(
            tune_s1_dict[s1_id],
            target_dict[cid],
            name_retrieval_score=n_score,
            addr_retrieval_score=a_score,
            num_blocking_methods=num_m
        )
        tune_records.append(feats)
        tune_meta.append((s1_id, cid))

    tune_X = pd.DataFrame(tune_records)[FEATURE_NAMES]

    # Load existing baseline model
    old_model = joblib.load(models_dir / "matcher.joblib")

    # Predict with both
    old_probs = old_model.predict_proba(tune_X)[:, 1]
    new_probs = new_model.predict_proba(tune_X)[:, 1]

    # Candidate Recall on Tuning Queries
    tune_s1_ids = list(india_tune["entity_id"])
    total_true_links = sum(len(gt_map[s1]) for s1 in tune_s1_ids)
    retrieved_true_links = 0
    cand_pairs_set = {(s1, cid) for s1, cid in tune_meta}
    for s1 in tune_s1_ids:
        for tid in gt_map[s1]:
            if (s1, tid) in cand_pairs_set:
                retrieved_true_links += 1
    cand_recall = retrieved_true_links / max(1, total_true_links)

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON ON FULL 4.13M INDIA TARGET INDEX (50 TUNE QUERIES):")
    print(f"Total True Links: {total_true_links} | Candidate Recall: {cand_recall*100:.2f}% ({retrieved_true_links}/{total_true_links})")
    print("=" * 80)

    # Compare at different thresholds
    print(f"{'Threshold':<10} | {'Model':<15} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<14} | {'Avg Matches':<12}")
    print("-" * 90)

    for thr in [0.50, 0.65, 0.75, 0.85]:
        # Old Model
        old_preds = {eid: set() for eid in tune_s1_ids}
        for (s1, cid), p in zip(tune_meta, old_probs):
            if p >= thr:
                old_preds[s1].add(cid)
        old_eval = evaluate_predictions(old_preds, gt_map, tune_s1_ids)

        # New Model
        new_preds = {eid: set() for eid in tune_s1_ids}
        for (s1, cid), p in zip(tune_meta, new_probs):
            if p >= thr:
                new_preds[s1].add(cid)
        new_eval = evaluate_predictions(new_preds, gt_map, tune_s1_ids)

        print(f"{thr:<10.2f} | {'Old (Curated)':<15} | {old_eval['macro_f0_5']:<12.4f} | {old_eval['macro_precision']:<10.4f} | {old_eval['macro_recall']:<10.4f} | {old_eval['singleton_accuracy']*100:<13.1f}% | {old_eval['avg_predicted_matches']:<12.2f}")
        print(f"{thr:<10.2f} | {'New (Realistic)':<15} | {new_eval['macro_f0_5']:<12.4f} | {new_eval['macro_precision']:<10.4f} | {new_eval['macro_recall']:<10.4f} | {new_eval['singleton_accuracy']*100:<13.1f}% | {new_eval['avg_predicted_matches']:<12.2f}")
        print("-" * 90)


if __name__ == "__main__":
    run_hard_negatives_experiment()
