"""Benchmark representative tuning queries against the FULL uncurated training target index.

Evaluates blocking recall and candidate distributions without any ground-truth
filtering or favorable distractor pool construction.
"""

import sys
from pathlib import Path
import json
import time
import gc
import psutil
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from src.data_loader import get_default_paths, iter_source_tsv, load_ground_truth
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker
from joblib import Parallel, delayed


def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_country_full_index(
    country: str,
    s1_tune_df: pd.DataFrame,
    gt_map: dict,
    max_targets: int = None
):
    paths = get_default_paths()
    print(f"\n{'='*60}")
    print(f"BENCHMARKING FULL TARGET INDEX FOR COUNTRY: {country}")
    print(f"{'='*60}")
    print(f"Tuning queries for {country}: {len(s1_tune_df):,}")

    t0 = time.time()
    mem_start = get_memory_usage_mb()

    # 1. Stream full target records for this country from S2 and S3
    print(f"Streaming full {country} target records from train_source2.tsv and train_source3.tsv...")
    s2_chunks = []
    for chunk in iter_source_tsv(paths["train_source2"], chunksize=250000):
        hit = chunk[chunk["country"] == country][["entity_id", "business_name", "business_address", "country"]]
        if len(hit) > 0:
            s2_chunks.append(hit)
        if max_targets and sum(len(h) for h in s2_chunks) >= max_targets // 2:
            break

    s3_chunks = []
    for chunk in iter_source_tsv(paths["train_source3"], chunksize=250000):
        hit = chunk[chunk["country"] == country][["entity_id", "business_name", "business_address", "country"]]
        if len(hit) > 0:
            s3_chunks.append(hit)
        if max_targets and sum(len(h) for h in s3_chunks) >= max_targets // 2:
            break

    targets_df = pd.concat(s2_chunks + s3_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    del s2_chunks, s3_chunks
    gc.collect()

    t_load = time.time() - t0
    mem_loaded = get_memory_usage_mb()
    print(f"Loaded {len(targets_df):,} full target records in {t_load:.1f}s (RAM: {mem_loaded:.1f} MB, Δ: {mem_loaded - mem_start:.1f} MB).")

    # 2. Normalization
    print("Normalizing names and addresses in parallel...")
    t_norm_0 = time.time()
    t_names = targets_df["business_name"].fillna("").astype(str).tolist()
    t_addrs = targets_df["business_address"].fillna("").astype(str).tolist()
    
    chunk_size = 20000
    chunks = [(t_names[i:i+chunk_size], t_addrs[i:i+chunk_size]) for i in range(0, len(t_names), chunk_size)]
    
    def _norm_chunk(b_names, b_addrs):
        return [normalize_name(n)[1] for n in b_names], [normalize_address(a)[0] for a in b_addrs]

    norm_res = Parallel(n_jobs=8, batch_size=4)(delayed(_norm_chunk)(cn, ca) for cn, ca in chunks)
    targets_df["name_core"] = [c for r in norm_res for c in r[0]]
    targets_df["addr_norm"] = [a for r in norm_res for a in r[1]]
    del t_names, t_addrs, chunks, norm_res
    gc.collect()

    s1_tune_df = s1_tune_df.copy()
    s1_tune_df["name_core"] = [normalize_name(n)[1] for n in s1_tune_df["business_name"]]
    s1_tune_df["addr_norm"] = [normalize_address(a)[0] for a in s1_tune_df["business_address"]]
    t_norm = time.time() - t_norm_0
    print(f"Normalized records in {t_norm:.1f}s.")

    # 3. Fit Blocker
    print("Fitting CountryBlocker on full target index...")
    t_fit_0 = time.time()
    blocker = CountryBlocker(
        country=country,
        name_top_k=25,
        addr_top_k=8,
        name_sim_threshold=0.10,
        addr_sim_threshold=0.18
    )
    blocker.fit(targets_df)
    t_fit = time.time() - t_fit_0
    mem_fit = get_memory_usage_mb()
    print(f"Fitted blocker in {t_fit:.1f}s (RAM: {mem_fit:.1f} MB, Peak Δ: {mem_fit - mem_start:.1f} MB).")

    # 4. Query Blocker
    print(f"Querying {len(s1_tune_df)} tuning queries...")
    t_query_0 = time.time()
    cands = blocker.query(s1_tune_df, batch_size=250)
    t_query = time.time() - t_query_0

    # 5. Evaluate Blocking Recall and Candidate Distribution
    cand_counts_by_s1 = {eid: 0 for eid in s1_tune_df["entity_id"]}
    cand_set = set()
    for item in cands:
        s1_id, cid = item[0], item[1]
        cand_counts_by_s1[s1_id] += 1
        cand_set.add((s1_id, cid))

    counts = list(cand_counts_by_s1.values())
    total_true_links = 0
    retrieved_true_links = 0
    for s1 in s1_tune_df["entity_id"]:
        true_targets = gt_map.get(s1, set())
        for tid in true_targets:
            total_true_links += 1
            if (s1, tid) in cand_set:
                retrieved_true_links += 1

    recall = (retrieved_true_links / total_true_links) if total_true_links > 0 else 1.0

    stats = {
        "country": country,
        "queries_evaluated": len(s1_tune_df),
        "indexed_targets_count": len(targets_df),
        "candidate_pairs_generated": len(cands),
        "candidate_counts": {
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95))
        },
        "blocking_recall": {
            "true_links_total": total_true_links,
            "true_links_retrieved": retrieved_true_links,
            "recall": float(recall)
        },
        "timing_seconds": {
            "target_stream_load": float(t_load),
            "normalization": float(t_norm),
            "indexing_fit": float(t_fit),
            "candidate_query": float(t_query),
            "total": float(time.time() - t0)
        },
        "memory_mb": {
            "start": float(mem_start),
            "after_load": float(mem_loaded),
            "after_fit": float(mem_fit),
            "peak_delta": float(mem_fit - mem_start)
        }
    }

    print("\nBenchmark Results:")
    print(f"  Indexed Targets:         {stats['indexed_targets_count']:,}")
    print(f"  Candidate Pairs:         {stats['candidate_pairs_generated']:,} ({stats['candidate_counts']['mean']:.1f} per query, median {stats['candidate_counts']['median']:.1f})")
    print(f"  Candidate Range:         [{stats['candidate_counts']['min']}, {stats['candidate_counts']['max']}] (P95: {stats['candidate_counts']['p95']:.1f})")
    print(f"  Blocking Recall:         {stats['blocking_recall']['recall']:.4f} ({stats['blocking_recall']['true_links_retrieved']:,}/{stats['blocking_recall']['true_links_total']:,})")
    print(f"  Target Fit Time:         {stats['timing_seconds']['indexing_fit']:.1f}s")
    print(f"  Query Time:              {stats['timing_seconds']['candidate_query']:.2f}s ({stats['timing_seconds']['candidate_query']/len(s1_tune_df)*1000:.1f} ms/query)")
    print(f"  Peak Memory Delta:       {stats['memory_mb']['peak_delta']:.1f} MB")

    # Cleanup memory
    del blocker, targets_df, cands
    gc.collect()

    return stats


def main():
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"
    paths = get_default_paths()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Use Tune queries only
    tune_ids = set(manifest["tune_ids"])
    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    tune_s1_df = s1_df[s1_df["entity_id"].isin(tune_ids)].copy().reset_index(drop=True)

    # Load ground truth for tune queries
    print("Loading Ground Truth for Tune queries...")
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(tune_ids)]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}
    for s1 in tune_ids:
        if s1 not in gt_map:
            gt_map[s1] = set()

    # Representative slice of tuning queries: 50 India queries, 50 US queries
    india_tune = tune_s1_df[tune_s1_df["country"] == "India"].head(50)
    us_tune = tune_s1_df[tune_s1_df["country"] == "US"].head(50)

    results = {}

    # Run India first (India has ~4.1M targets)
    india_stats = benchmark_country_full_index("India", india_tune, gt_map)
    results["India"] = india_stats

    # Run US (US has ~6.2M targets)
    us_stats = benchmark_country_full_index("US", us_tune, gt_map)
    results["US"] = us_stats

    out_file = PROJECT_ROOT / "experiments" / "full_index_tune_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull-index benchmark results saved to {out_file}")


if __name__ == "__main__":
    main()
