"""Train matcher on 25k query sample with blocker-mined hard negatives and tune Macro F0.5."""

import sys
from pathlib import Path
import os
import json
import time
import logging
import gc
import re
import pandas as pd
import numpy as np
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.data_loader import get_default_paths, load_ground_truth, iter_source_tsv
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker
from src.features import FEATURE_NAMES, compute_pair_features, enrich_record_dict
from src.evaluate import evaluate_predictions, compute_entity_metrics


def parallel_normalize(df: pd.DataFrame, n_jobs: int = 8, chunk_size: int = 20000) -> pd.DataFrame:
    """Normalize names and addresses in parallel chunks."""
    names = df["business_name"].fillna("").astype(str).tolist()
    addrs = df["business_address"].fillna("").astype(str).tolist()

    chunks = [(names[i:i + chunk_size], addrs[i:i + chunk_size]) for i in range(0, len(names), chunk_size)]

    def _worker(b_n, b_a):
        return [normalize_name(n)[1] for n in b_n], [normalize_address(a)[0] for a in b_a]

    res = Parallel(n_jobs=n_jobs, batch_size=4)(delayed(_worker)(cn, ca) for cn, ca in chunks)
    df = df.copy()
    df["name_core"] = [c for r in res for c in r[0]]
    df["addr_norm"] = [a for r in res for a in r[1]]
    return df


def main():
    start_time = time.time()
    paths = get_default_paths()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = PROJECT_ROOT / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1 & 2: HARD-NEGATIVE DATASET GENERATION & GBDT RETRAINING")
    print("=" * 70)

    # 1. Load Ground Truth
    print("\n[1/7] Loading full training ground truth...")
    gt_df = load_ground_truth(paths["train_gt"])
    print(f"  Loaded {len(gt_df):,} ground truth rows.")

    gt_map = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}
        else:
            gt_map[s1] = set()

    # 2. Stratified Sampling of 25,000 S1 queries (15k US, 10k India, 5.58% singletons)
    print("\n[2/7] Sampling 25,000 S1 queries stratified across countries & cardinality...")
    s1_us_list = []
    s1_in_list = []

    # Stream S1 to collect candidate pools
    for chunk in iter_source_tsv(paths["train_source1"], chunksize=250000):
        us_c = chunk[chunk["country"] == "US"]
        in_c = chunk[chunk["country"] == "India"]
        if len(s1_us_list) * 250000 < 100000:
            s1_us_list.append(us_c)
        if len(s1_in_list) * 250000 < 100000:
            s1_in_list.append(in_c)
        if sum(len(c) for c in s1_us_list) >= 40000 and sum(len(c) for c in s1_in_list) >= 30000:
            break

    s1_us_pool = pd.concat(s1_us_list, ignore_index=True)
    s1_in_pool = pd.concat(s1_in_list, ignore_index=True)

    def stratify_sample(pool_df: pd.DataFrame, target_n: int, seed: int = 42) -> pd.DataFrame:
        cats = []
        for eid in pool_df["entity_id"]:
            mc = len(gt_map.get(eid, set()))
            if mc == 0:
                cats.append("sing_0")
            elif mc == 1:
                cats.append("match_1")
            elif mc <= 3:
                cats.append("match_2_3")
            else:
                cats.append("match_4_plus")
        pool_df = pool_df.copy()
        pool_df["strata"] = cats
        sss = StratifiedShuffleSplit(n_splits=1, train_size=target_n, random_state=seed)
        idx, _ = next(sss.split(pool_df, pool_df["strata"]))
        return pool_df.iloc[idx].copy().reset_index(drop=True)

    s1_us_sample = stratify_sample(s1_us_pool, 15000, seed=42)
    s1_in_sample = stratify_sample(s1_in_pool, 10000, seed=42)
    s1_all = pd.concat([s1_us_sample, s1_in_sample], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Sampled {len(s1_all):,} queries: {len(s1_us_sample):,} US, {len(s1_in_sample):,} India.")
    singletons = sum(1 for eid in s1_all["entity_id"] if len(gt_map.get(eid, set())) == 0)
    print(f"  Sampled Singletons: {singletons:,} ({singletons / len(s1_all) * 100:.2f}%)")

    # Split into Train (70% = 17,500), Tune (15% = 3,750), Assessment (15% = 3,750)
    s1_all["split_strata"] = s1_all["country"] + "__" + s1_all["strata"]
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    tr_idx, rem_idx = next(sss1.split(s1_all, s1_all["split_strata"]))
    train_df = s1_all.iloc[tr_idx].copy().reset_index(drop=True)
    rem_df = s1_all.iloc[rem_idx].copy().reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    tu_idx, as_idx = next(sss2.split(rem_df, rem_df["split_strata"]))
    tune_df = rem_df.iloc[tu_idx].copy().reset_index(drop=True)
    assess_df = rem_df.iloc[as_idx].copy().reset_index(drop=True)

    train_ids = set(train_df["entity_id"])
    tune_ids = set(tune_df["entity_id"])
    assess_ids = set(assess_df["entity_id"])

    print(f"  Split counts: Train={len(train_ids):,}, Tune={len(tune_ids):,}, Assessment={len(assess_ids):,}")

    # Needed true targets across the 25k queries
    needed_s2 = set()
    needed_s3 = set()
    for eid in s1_all["entity_id"]:
        for tid in gt_map.get(eid, set()):
            if tid.startswith("S2-"):
                needed_s2.add(tid)
            elif tid.startswith("S3-"):
                needed_s3.add(tid)
    print(f"  True targets to include in target pools: S2={len(needed_s2):,}, S3={len(needed_s3):,}")

    # 3. Build target pools per country (True Targets + 250k Real Distractors per source)
    print("\n[3/7] Building target pools with true targets and 250,000 distractors per source...")
    all_cand_records = []

    for country in ["India", "US"]:
        s1_c = s1_all[s1_all["country"] == country].copy().reset_index(drop=True)
        print(f"\n--- Processing Country: {country} ({len(s1_c):,} queries) ---")

        # Stream S2
        s2_hits = []
        s2_dist = []
        dist_cap = 250000
        for chunk in iter_source_tsv(paths["train_source2"], chunksize=250000):
            c_part = chunk[chunk["country"] == country]
            if len(c_part) == 0:
                continue
            hit = c_part[c_part["entity_id"].isin(needed_s2)]
            if len(hit) > 0:
                s2_hits.append(hit)
            if sum(len(d) for d in s2_dist) < dist_cap:
                take = min(50000, dist_cap - sum(len(d) for d in s2_dist))
                dist = c_part[~c_part["entity_id"].isin(needed_s2)].head(take)
                s2_dist.append(dist)
            if sum(len(h) for h in s2_hits) >= len([tid for tid in needed_s2 if tid.startswith("S2-")]) and sum(len(d) for d in s2_dist) >= dist_cap:
                break

        # Stream S3
        s3_hits = []
        s3_dist = []
        for chunk in iter_source_tsv(paths["train_source3"], chunksize=250000):
            c_part = chunk[chunk["country"] == country]
            if len(c_part) == 0:
                continue
            hit = c_part[c_part["entity_id"].isin(needed_s3)]
            if len(hit) > 0:
                s3_hits.append(hit)
            if sum(len(d) for d in s3_dist) < dist_cap:
                take = min(50000, dist_cap - sum(len(d) for d in s3_dist))
                dist = c_part[~c_part["entity_id"].isin(needed_s3)].head(take)
                s3_dist.append(dist)
            if sum(len(h) for h in s3_hits) >= len([tid for tid in needed_s3 if tid.startswith("S3-")]) and sum(len(d) for d in s3_dist) >= dist_cap:
                break

        target_c = pd.concat(s2_hits + s2_dist + s3_hits + s3_dist, ignore_index=True).drop_duplicates(subset=["entity_id"])
        print(f"  {country} Target Pool: {len(target_c):,} records ({sum(len(h) for h in s2_hits+s3_hits):,} true targets, remainder distractors).")

        # Parallel Normalization
        print(f"  Normalizing {country} targets & queries in parallel...")
        target_c = parallel_normalize(target_c, n_jobs=8)
        s1_c = parallel_normalize(s1_c, n_jobs=8)

        # Fit Blocker
        print(f"  Fitting CountryBlocker on {country} targets...")
        t_fit_0 = time.time()
        blocker = CountryBlocker(country=country)
        blocker.fit(target_c)
        print(f"  Fitted blocker in {time.time() - t_fit_0:.1f}s.")

        # Query candidates in chunks
        print(f"  Retrieving candidates for {len(s1_c):,} {country} queries...")
        t_q_0 = time.time()
        c_tuples = blocker.query(s1_c)
        print(f"  Retrieved {len(c_tuples):,} candidates in {time.time() - t_q_0:.1f}s ({len(c_tuples)/len(s1_c):.1f} cands/query).")

        # Target lookup dictionary
        target_dict = {
            row["entity_id"]: enrich_record_dict(row)
            for _, row in target_c.iterrows()
        }
        s1_dict = {
            row["entity_id"]: enrich_record_dict(row)
            for _, row in s1_c.iterrows()
        }

        # Build feature rows
        print(f"  Computing pairwise features for {country} candidate pairs...")
        t_f_0 = time.time()
        for item in c_tuples:
            s1_id = item[0]
            cid = item[1]
            n_score = item[3]
            a_score = item[4]
            num_m = item[6]

            if s1_id not in s1_dict or cid not in target_dict:
                continue

            feats = compute_pair_features(
                s1_dict[s1_id],
                target_dict[cid],
                name_retrieval_score=n_score,
                addr_retrieval_score=a_score,
                num_blocking_methods=num_m
            )
            is_match = 1 if cid in gt_map.get(s1_id, set()) else 0

            split = "train" if s1_id in train_ids else ("tune" if s1_id in tune_ids else "assessment")
            row_dict = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "label": is_match,
                "split": split,
            }
            row_dict.update(feats)
            all_cand_records.append(row_dict)

        print(f"  Features extracted in {time.time() - t_f_0:.1f}s.")

        del target_c, s1_c, blocker, target_dict, s1_dict, c_tuples, s2_hits, s2_dist, s3_hits, s3_dist
        gc.collect()

    pair_df = pd.DataFrame(all_cand_records)
    print(f"\nTotal Dataset Created: {len(pair_df):,} candidate pairs.")
    print(f"  Positive pairs (label=1): {int((pair_df['label'] == 1).sum()):,}")
    print(f"  Negative pairs (label=0): {int((pair_df['label'] == 0).sum()):,}")
    print(f"  Class balance: 1 : {(pair_df['label'] == 0).sum() / max(1, (pair_df['label'] == 1).sum()):.1f} (real-world hard negatives)")

    # 4. Train Model (Step 2)
    print("\n[4/7] Training HistGradientBoostingClassifier on Train Split...")
    train_pairs = pair_df[pair_df["split"] == "train"]
    tune_pairs = pair_df[pair_df["split"] == "tune"]
    assess_pairs = pair_df[pair_df["split"] == "assessment"]

    print(f"  Train Pairs:      {len(train_pairs):,} (from {len(train_ids):,} S1 queries)")
    print(f"  Tune Pairs:       {len(tune_pairs):,} (from {len(tune_ids):,} S1 queries)")
    print(f"  Assessment Pairs: {len(assess_pairs):,} (from {len(assess_ids):,} S1 queries)")

    X_train = train_pairs[FEATURE_NAMES]
    y_train = train_pairs["label"]

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.08,
        l2_regularization=0.5,
        random_state=42
    )

    t_train_0 = time.time()
    clf.fit(X_train, y_train)
    print(f"  Model trained in {time.time() - t_train_0:.1f}s.")

    # 5. Threshold Tuning on Tune Split (Step 3: Accounting for real recall ceiling)
    print("\n[5/7] Tuning Decision Threshold for Macro F0.5 on Tune Split...")
    tune_probs = clf.predict_proba(tune_pairs[FEATURE_NAMES])[:, 1]
    tune_eval_df = tune_pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    tune_eval_df["prob"] = tune_probs

    best_thresh = 0.50
    best_f05 = -1.0
    best_res = {}

    threshold_grid = np.linspace(0.40, 0.90, 26)
    print(f"  Grid searching {len(threshold_grid)} thresholds from 0.40 to 0.90...")

    tune_s1_list = list(tune_ids)
    for t in threshold_grid:
        t_val = round(float(t), 2)
        # Form prediction map
        filtered = tune_eval_df[tune_eval_df["prob"] >= t_val]
        pred_map = {eid: set() for eid in tune_s1_list}
        for _, r in filtered.iterrows():
            pred_map[r["source1_entity_id"]].add(r["candidate_entity_id"])

        res = evaluate_predictions(pred_map, gt_map, tune_s1_list)
        if res["macro_f0_5_raw"] > best_f05:
            best_f05 = res["macro_f0_5_raw"]
            best_thresh = t_val
            best_res = res

    print("\nOptimal Threshold Found on Tune Split:")
    print(f"  Threshold:           {best_thresh}")
    print(f"  Tune Macro F0.5:     {best_res['macro_f0_5']:.4f}")
    print(f"  Tune Macro Precision:{best_res['macro_precision']:.4f}")
    print(f"  Tune Macro Recall:   {best_res['macro_recall']:.4f}")
    print(f"  Tune Singleton Acc:  {best_res['singleton_accuracy']}")

    # 6. Final Evaluation on Held-Out Assessment Split
    print("\n[6/7] Evaluating Frozen Model on Held-Out Assessment Split (3,750 queries)...")
    assess_probs = clf.predict_proba(assess_pairs[FEATURE_NAMES])[:, 1]
    assess_eval_df = assess_pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    assess_eval_df["prob"] = assess_probs

    filtered_assess = assess_eval_df[assess_eval_df["prob"] >= best_thresh]
    assess_pred_map = {eid: set() for eid in assess_ids}
    for _, r in filtered_assess.iterrows():
        assess_pred_map[r["source1_entity_id"]].add(r["candidate_entity_id"])

    assess_res = evaluate_predictions(assess_pred_map, gt_map, list(assess_ids))
    print(f"\nHeld-Out Assessment Metrics (Threshold={best_thresh}):")
    print(f"  Assessment Macro F0.5:      {assess_res['macro_f0_5']:.4f}")
    print(f"  Assessment Macro Precision: {assess_res['macro_precision']:.4f}")
    print(f"  Assessment Macro Recall:    {assess_res['macro_recall']:.4f}")
    print(f"  Assessment Singleton Acc:   {assess_res['singleton_accuracy']}")
    print(f"  Assessment Queries:         {assess_res['total_evaluated_s1']:,}")

    # 7. Save Model Artifacts
    print("\n[7/7] Saving model artifacts to models/...")
    model_path = models_dir / "matcher.joblib"
    config_path = models_dir / "model_config.json"

    joblib.dump(clf, model_path)
    model_config = {
        "model_type": "HistGradientBoostingClassifier",
        "threshold": best_thresh,
        "feature_names": FEATURE_NAMES,
        "tune_macro_f05": best_res["macro_f0_5_raw"],
        "assessment_macro_f05": assess_res["macro_f0_5_raw"],
        "assessment_precision": assess_res["macro_precision_raw"],
        "assessment_recall": assess_res["macro_recall_raw"],
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    with open(exp_dir / "assessment_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(assess_res, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nHard-Negative Retraining Complete in {total_time/60:.1f} minutes!")


if __name__ == "__main__":
    main()
