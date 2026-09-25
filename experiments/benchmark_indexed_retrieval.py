"""Benchmark fast multi-key inverted index on 4.1M India targets."""

import sys
from pathlib import Path
import json
import time
import gc
import psutil
from collections import defaultdict
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_name, normalize_address, extract_postal_code
from src.data_loader import get_default_paths, iter_source_tsv, load_ground_truth

STOPWORDS = {
    "pvt", "ltd", "inc", "llc", "corp", "corporation", "private", "limited",
    "the", "and", "co", "company", "group", "enterprises", "solutions",
    "services", "technologies", "tech", "trading", "retail", "international"
}

def extract_blocking_keys(name_norm: str, addr_norm: str, country: str = "India"):
    keys = set()
    words = [w for w in name_norm.split() if len(w) >= 3 and w not in STOPWORDS]

    # 1. Exact full normalized core
    if name_norm.strip():
        keys.add(("core", name_norm.strip()))

    # 2. First 2 significant words
    if len(words) >= 2:
        keys.add(("prefix2", f"{words[0]}_{words[1]}"))

    # 3. Individual rare words
    for w in words[:3]:
        keys.add(("word", w))

    # 4. Postal code + first word
    pin = extract_postal_code(addr_norm, country)
    if pin and words:
        keys.add(("pin_word", f"{pin}_{words[0]}"))

    # 5. Character 3-grams of significant words
    for w in words:
        if len(w) >= 3:
            for i in range(len(w) - 2):
                keys.add(("ng3", w[i:i+3]))

    return keys

def main():
    paths = get_default_paths()
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tune_ids = set(manifest["tune_ids"])
    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    india_tune = s1_df[(s1_df["entity_id"].isin(tune_ids)) & (s1_df["country"] == "India")].head(50)

    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(set(india_tune["entity_id"]))]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}

    total_true = sum(len(v) for v in gt_map.values())
    print(f"Testing 50 India tune queries with {total_true} true ground-truth links.")

    # 1. Stream and build inverted index for India (from S2 and S3)
    print("Streaming and indexing full India target records...")
    t0 = time.time()
    
    # Store targets compactly
    target_ids = []
    target_names = []
    target_addrs = []
    index = defaultdict(list)
    key_counts = defaultdict(int)

    count = 0
    for src_path, src_label in [(paths["train_source2"], "S2"), (paths["train_source3"], "S3")]:
        for chunk in iter_source_tsv(src_path, chunksize=250000):
            hit = chunk[chunk["country"] == "India"][["entity_id", "business_name", "business_address"]]
            eids = hit["entity_id"].values
            bnames = hit["business_name"].fillna("").values
            baddrs = hit["business_address"].fillna("").values
            for eid, bname, baddr in zip(eids, bnames, baddrs):
                n_norm = normalize_name(bname)[1]
                a_norm = normalize_address(baddr)[0]

                idx = count
                target_ids.append(eid)
                target_names.append(n_norm)
                target_addrs.append(a_norm)
                count += 1

                for k in extract_blocking_keys(n_norm, a_norm, country="India"):
                    if key_counts[k] < 500:
                        index[k].append(idx)
                        key_counts[k] += 1

    t_build = time.time() - t0
    mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Indexed {count:,} targets into {len(index):,} keys in {t_build:.1f}s. Memory RSS: {mem_mb:.1f} MB.")

    # 2. Query the 50 queries
    print("Querying 50 tune queries...")
    t_query_0 = time.time()
    retrieved_true = 0
    candidate_counts = []

    for _, q in india_tune.iterrows():
        s1 = q["entity_id"]
        q_name = normalize_name(q["business_name"])[1]
        q_addr = normalize_address(q["business_address"])[0]

        q_keys = extract_blocking_keys(q_name, q_addr)
        cand_indices = set()
        for k in q_keys:
            if k in index:
                cand_indices.update(index[k])

        # Rank candidates by name fuzz ratio and keep top-35
        cand_scores = []
        for c_idx in cand_indices:
            score = fuzz.ratio(q_name, target_names[c_idx])
            cand_scores.append((score, c_idx))

        cand_scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [target_ids[c_idx] for score, c_idx in cand_scores[:35]]

        candidate_counts.append(len(top_candidates))
        true_set = gt_map.get(s1, set())
        for tid in true_set:
            if tid in top_candidates:
                retrieved_true += 1

    t_query = time.time() - t_query_0
    recall = retrieved_true / total_true if total_true > 0 else 1.0

    print("\nRESULTS:")
    print(f"  Recall on true links: {recall:.4f} ({retrieved_true}/{total_true})")
    print(f"  Average candidates per query: {np.mean(candidate_counts):.1f}")
    print(f"  Query time for 50 queries: {t_query:.3f}s ({t_query/50*1000:.2f} ms/query)")
    print(f"  Throughput: {50 / t_query:.1f} queries/sec")

if __name__ == "__main__":
    main()
