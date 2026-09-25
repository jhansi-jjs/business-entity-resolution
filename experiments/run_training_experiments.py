"""End-to-end training, validation, ablation, and assessment pipeline with reproducible splits."""

import sys
from pathlib import Path
import json
import logging
import gc
import time
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth, iter_source_tsv
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker
from src.features import FEATURE_NAMES, enrich_record_dict
from src.evaluate import evaluate_predictions, evaluate_predictions_detailed
from src.train import build_pair_dataset, train_and_compare_models, tune_threshold_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    paths = get_default_paths()
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"

    if not manifest_path.exists() or not sampled_s1_path.exists():
        raise FileNotFoundError(
            f"Split manifest or sampled S1 not found! Run experiments/create_splits.py first."
        )

    print("\n" + "="*70)
    print("STARTING REPRODUCIBLE BENCHMARK EXPERIMENTS")
    print("="*70)

    # 1. Load splits
    print("\n[1/7] Loading split manifest and sampled S1 records...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_ids = set(manifest["train_ids"])
    tune_ids = set(manifest["tune_ids"])
    assessment_ids = set(manifest["assessment_ids"])
    all_s1_ids = list(train_ids | tune_ids | assessment_ids)

    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    print(f"  Loaded {len(s1_df):,} sampled S1 queries:")
    print(f"    Train:      {len(train_ids):,} ({len(train_ids)/len(all_s1_ids)*100:.1f}%)")
    print(f"    Tune:       {len(tune_ids):,} ({len(tune_ids)/len(all_s1_ids)*100:.1f}%)")
    print(f"    Assessment: {len(assessment_ids):,} ({len(assessment_ids)/len(all_s1_ids)*100:.1f}%)")

    # 2. Load ground truth for these S1 IDs
    print("\n[2/7] Loading ground truth mapping for sampled S1 queries...")
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(set(all_s1_ids))]

    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()

    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            m_set = {m.strip() for m in raw_m.split(",") if m.strip()}
            gt_map[s1] = m_set
            for mid in m_set:
                if mid.startswith("S2-"):
                    needed_s2.add(mid)
                elif mid.startswith("S3-"):
                    needed_s3.add(mid)

    # Ensure singletons are explicitly in gt_map as empty set
    for s1 in all_s1_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()

    total_true_links = sum(len(v) for v in gt_map.values())
    singletons_count = sum(1 for v in gt_map.values() if len(v) == 0)
    print(f"  Sampled S1 with >= 1 matches: {len(all_s1_ids) - singletons_count:,}")
    print(f"  Sampled S1 singletons (0 matches): {singletons_count:,}")
    print(f"  Total true links to find: {total_true_links:,}")
    print(f"  Unique target IDs needed: S2={len(needed_s2):,}, S3={len(needed_s3):,}")

    # 3. Stream S2 and S3 for target records + distractors
    print("\n[3/7] Scanning S2 and S3 for true target entities + distractor pool...")
    distractor_cap_per_source = 10000

    # Scan S2
    s2_hits = []
    s2_distractors = []
    s2_distractor_count = 0
    t0 = time.time()

    for chunk in iter_source_tsv(paths["train_source2"], chunksize=250000):
        hit = chunk[chunk["entity_id"].isin(needed_s2)]
        if len(hit) > 0:
            s2_hits.append(hit)
        if s2_distractor_count < distractor_cap_per_source:
            to_take = min(2500, distractor_cap_per_source - s2_distractor_count)
            dist = chunk[~chunk["entity_id"].isin(needed_s2)].head(to_take)
            s2_distractors.append(dist)
            s2_distractor_count += len(dist)
        found_now = sum(len(h) for h in s2_hits)
        if found_now >= len(needed_s2) and s2_distractor_count >= distractor_cap_per_source:
            print(f"  Found all {found_now:,} S2 targets early!")
            break

    s2_targets = pd.concat(s2_hits + s2_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"  S2 Target Pool: {len(s2_targets):,} records ({sum(len(h) for h in s2_hits):,} true targets, {s2_distractor_count:,} distractors) in {time.time()-t0:.1f}s.")

    # Scan S3
    s3_hits = []
    s3_distractors = []
    s3_distractor_count = 0
    t0 = time.time()

    for chunk in iter_source_tsv(paths["train_source3"], chunksize=250000):
        hit = chunk[chunk["entity_id"].isin(needed_s3)]
        if len(hit) > 0:
            s3_hits.append(hit)
        if s3_distractor_count < distractor_cap_per_source:
            to_take = min(2500, distractor_cap_per_source - s3_distractor_count)
            dist = chunk[~chunk["entity_id"].isin(needed_s3)].head(to_take)
            s3_distractors.append(dist)
            s3_distractor_count += len(dist)
        found_now = sum(len(h) for h in s3_hits)
        if found_now >= len(needed_s3) and s3_distractor_count >= distractor_cap_per_source:
            print(f"  Found all {found_now:,} S3 targets early!")
            break

    s3_targets = pd.concat(s3_hits + s3_distractors, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"  S3 Target Pool: {len(s3_targets):,} records ({sum(len(h) for h in s3_hits):,} true targets, {s3_distractor_count:,} distractors) in {time.time()-t0:.1f}s.")

    target_df = pd.concat([s2_targets, s3_targets], ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"  Total target search space: {len(target_df):,} records.")

    # 4. Normalize and Run Blocking
    print("\n[4/7] Normalizing records and generating candidate pairs...")
    s1_df["name_core"] = [normalize_name(n)[1] for n in s1_df["business_name"]]
    s1_df["addr_norm"] = [normalize_address(a)[0] for a in s1_df["business_address"]]
    target_df["name_core"] = [normalize_name(n)[1] for n in target_df["business_name"]]
    target_df["addr_norm"] = [normalize_address(a)[0] for a in target_df["business_address"]]

    all_candidates = []
    blocking_recall_by_country = {}

    for country in ["US", "India"]:
        q_c = s1_df[s1_df["country"] == country]
        t_c = target_df[target_df["country"] == country]
        if len(q_c) == 0 or len(t_c) == 0:
            continue
        print(f"  Blocking {country}: {len(q_c):,} queries vs {len(t_c):,} targets...")
        blocker = CountryBlocker(
            country=country,
            name_top_k=25,
            addr_top_k=8,
            name_sim_threshold=0.10,
            addr_sim_threshold=0.18
        )
        blocker.fit(t_c)
        cands = blocker.query(q_c, batch_size=500)
        all_candidates.extend(cands)

        # Calculate blocking recall for this country
        c_s1_ids = set(q_c["entity_id"])
        c_true_pairs = set()
        for s1 in c_s1_ids:
            for tid in gt_map.get(s1, set()):
                c_true_pairs.add((s1, tid))

        c_retrieved_true = 0
        cand_pairs_set = {(item[0], item[1]) for item in cands}
        for s1, tid in c_true_pairs:
            if (s1, tid) in cand_pairs_set:
                c_retrieved_true += 1

        rec = (c_retrieved_true / len(c_true_pairs)) if len(c_true_pairs) > 0 else 1.0
        blocking_recall_by_country[country] = {
            "queries": len(q_c),
            "targets": len(t_c),
            "candidate_pairs": len(cands),
            "candidates_per_query": len(cands) / len(q_c),
            "true_pairs_in_ground_truth": len(c_true_pairs),
            "true_pairs_retrieved": c_retrieved_true,
            "blocking_recall": rec
        }
        print(f"    {country} candidate pairs: {len(cands):,} ({len(cands)/len(q_c):.1f} per query)")
        print(f"    {country} blocking recall: {rec:.4f} ({c_retrieved_true:,}/{len(c_true_pairs):,})")

    # Save blocking recall stats
    blocking_stats_file = PROJECT_ROOT / "experiments" / "blocking_recall_benchmark.json"
    with open(blocking_stats_file, "w", encoding="utf-8") as f:
        json.dump(blocking_recall_by_country, f, indent=2)
    print(f"  Blocking benchmark saved to {blocking_stats_file}")

    # 5. Build Pairwise Feature Dataset
    print("\n[5/7] Building pairwise feature dataset...")
    s1_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in s1_df.iterrows()}
    cand_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in target_df.iterrows()}

    pair_df = build_pair_dataset(s1_dict, cand_dict, all_candidates, gt_map)
    pos_count = int(pair_df["label"].sum())
    neg_count = len(pair_df) - pos_count
    print(f"  Total pairs: {len(pair_df):,} (Positives: {pos_count:,}, Negatives: {neg_count:,}, Pos Ratio: {pos_count/len(pair_df):.4f})")

    # Split pair_df strictly into Train, Tune, Assessment by source1_entity_id
    train_pair_df = pair_df[pair_df["source1_entity_id"].isin(train_ids)].copy().reset_index(drop=True)
    tune_pair_df = pair_df[pair_df["source1_entity_id"].isin(tune_ids)].copy().reset_index(drop=True)
    assess_pair_df = pair_df[pair_df["source1_entity_id"].isin(assessment_ids)].copy().reset_index(drop=True)

    print(f"  Split pair counts:")
    print(f"    Train pairs:      {len(train_pair_df):,} across {len(train_ids):,} S1 queries (Pos: {train_pair_df['label'].sum():,})")
    print(f"    Tune pairs:       {len(tune_pair_df):,} across {len(tune_ids):,} S1 queries (Pos: {tune_pair_df['label'].sum():,})")
    print(f"    Assessment pairs: {len(assess_pair_df):,} across {len(assessment_ids):,} S1 queries (Pos: {assess_pair_df['label'].sum():,})")

    # Save pair dataset for reproducibility
    pair_df.to_parquet = None  # in case
    pair_tsv_path = PROJECT_ROOT / "experiments" / "splits" / "benchmark_pairs.tsv"
    pair_df.to_csv(pair_tsv_path, sep="\t", index=False)
    print(f"  Pair dataset saved to {pair_tsv_path}")

    # 6. Train Models and Tune on Tune Set
    print("\n[6/7] Training models and tuning decision thresholds on Tune split...")
    models_dir = PROJECT_ROOT / "models"
    tune_s1_list = list(tune_ids)
    exp_summary = train_and_compare_models(
        train_df=train_pair_df,
        val_df=tune_pair_df,
        val_s1_ids=tune_s1_list,
        gt_map=gt_map,
        output_dir=models_dir
    )

    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS (TUNE SPLIT):")
    print("="*60)
    print(exp_summary.to_string(index=False))
    print("="*60)

    # 7. Evaluate Best Model ONCE on Held-out Assessment Split
    print("\n[7/7] Evaluating frozen best model on Held-Out Assessment split (500 S1 queries)...")
    best_model_path = models_dir / "matcher.joblib"
    config_path = models_dir / "model_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        m_config = json.load(f)

    best_clf = joblib.load(best_model_path)
    tuned_thresh = float(m_config["threshold"])
    assess_s1_list = list(assessment_ids)

    # Predict on assessment pairs
    assess_probs = best_clf.predict_proba(assess_pair_df[FEATURE_NAMES])[:, 1]
    assess_pred_df = assess_pair_df.copy()
    assess_pred_df["prob"] = assess_probs

    assess_pred_map = {s1: set() for s1 in assess_s1_list}
    filtered_assess = assess_pred_df[assess_pred_df["prob"] >= tuned_thresh]
    for s1, grp in filtered_assess.groupby("source1_entity_id"):
        assess_pred_map[s1] = set(grp["candidate_entity_id"])

    # Overall Assessment metrics
    assess_metrics = evaluate_predictions(assess_pred_map, gt_map, assess_s1_list)

    # Detailed evaluation with country and cardinality breakdowns
    assess_s1_df = s1_df[s1_df["entity_id"].isin(assessment_ids)].copy().reset_index(drop=True)
    detailed_report = evaluate_predictions_detailed(assess_pred_map, gt_map, assess_s1_df)

    assess_metrics = detailed_report["overall"]
    assess_by_country = detailed_report["by_country"]
    assess_by_cardinality = detailed_report["by_match_cardinality"]

    assessment_report = {
        "model": m_config["best_model"],
        "threshold": tuned_thresh,
        "sample_size": len(assess_s1_list),
        "overall_metrics": assess_metrics,
        "breakdowns": {
            "by_country": assess_by_country,
            "by_cardinality": assess_by_cardinality
        }
    }

    assess_report_file = PROJECT_ROOT / "experiments" / "assessment_evaluation.json"
    with open(assess_report_file, "w", encoding="utf-8") as f:
        json.dump(assessment_report, f, indent=2)

    print("\n" + "="*60)
    print("HELD-OUT ASSESSMENT EVALUATION REPORT:")
    print("="*60)
    print(f"  Model:              {m_config['best_model']}")
    print(f"  Decision Threshold: {tuned_thresh}")
    print(f"  Macro Precision:    {assess_metrics['macro_precision']}")
    print(f"  Macro Recall:       {assess_metrics['macro_recall']}")
    print(f"  Macro F0.5:         {assess_metrics['macro_f0_5']}")
    print(f"  Singleton Accuracy: {assess_metrics['singleton_accuracy']} ({assess_metrics['singleton_count']} singletons evaluated)")
    print(f"  Avg Predicted M:    {assess_metrics['avg_predicted_matches']}")

    print("\nBreakdown by Country on Assessment Set:")
    for cty, m in assess_by_country.items():
        print(f"  {cty}: F0.5={m['macro_f0_5']}, Prec={m['macro_precision']}, Rec={m['macro_recall']}, Singletons={m['singleton_accuracy']} (N={m['total_evaluated_s1']})")

    print("\nBreakdown by Match Cardinality on Assessment Set:")
    for cat, m in assess_by_cardinality.items():
        print(f"  {cat}: F0.5={m['macro_f0_5']}, Prec={m['macro_precision']}, Rec={m['macro_recall']}, N={m['total_evaluated_s1']}")

    print("="*60)
    print(f"Assessment report saved to {assess_report_file}")


if __name__ == "__main__":
    main()
