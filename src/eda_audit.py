"""Comprehensive EDA script for Business Entity Resolution Challenge.

Processes train and test datasets sequentially with memory-efficient chunking,
collecting all required structural diagnostics and saving eda_report.json and eda_report.md.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.data_loader import get_default_paths, iter_source_tsv, load_ground_truth


def audit_source_file(path: Path, source_name: str, expected_prefix: str) -> Dict[str, Any]:
    print(f"Auditing {source_name} ({path.name})...")
    total_rows = 0
    null_name = 0
    null_addr = 0
    null_country = 0
    empty_name = 0
    empty_addr = 0
    empty_country = 0
    
    country_counts = Counter()
    name_lengths = []
    addr_lengths = []
    
    id_set = set()
    dup_id_count = 0
    prefix_errors = 0
    
    sample_rows = []
    
    # Process in chunks
    for chunk in iter_source_tsv(path, chunksize=250000):
        if total_rows == 0:
            sample_rows = chunk.head(5).to_dict(orient="records")
            
        nrows = len(chunk)
        total_rows += nrows
        
        # Check IDs
        ids = chunk["entity_id"].astype(str)
        # Prefix check
        bad_pref = ~ids.str.startswith(expected_prefix)
        prefix_errors += int(bad_pref.sum())
        
        # Uniqueness check
        for eid in ids:
            if eid in id_set:
                dup_id_count += 1
            else:
                id_set.add(eid)
                
        # Missing & empty checks
        name_str = chunk["business_name"].fillna("").astype(str)
        addr_str = chunk["business_address"].fillna("").astype(str)
        cntry_str = chunk["country"].fillna("").astype(str)
        
        null_name += int((name_str == "nan").sum() + (name_str == "None").sum() + (name_str == "null").sum())
        empty_name += int((name_str.str.strip() == "").sum())
        
        null_addr += int((addr_str == "nan").sum() + (addr_str == "None").sum() + (addr_str == "null").sum())
        empty_addr += int((addr_str.str.strip() == "").sum())
        
        null_country += int((cntry_str == "nan").sum() + (cntry_str == "None").sum() + (cntry_str == "null").sum())
        empty_country += int((cntry_str.str.strip() == "").sum())
        
        # Country distribution
        for c, count in cntry_str.value_counts().items():
            country_counts[str(c)] += int(count)
            
        # Sample lengths for quantiles (sample 5,000 per chunk to keep memory minimal)
        step = max(1, nrows // 5000)
        name_lengths.extend(name_str.iloc[::step].str.len().tolist())
        addr_lengths.extend(addr_str.iloc[::step].str.len().tolist())

    name_arr = np.array(name_lengths)
    addr_arr = np.array(addr_lengths)
    
    return {
        "source_name": source_name,
        "filename": path.name,
        "total_rows": total_rows,
        "unique_entity_ids": len(id_set),
        "duplicate_entity_ids": dup_id_count,
        "prefix_errors": prefix_errors,
        "missing_or_empty_name": empty_name + null_name,
        "missing_or_empty_addr": empty_addr + null_addr,
        "missing_or_empty_country": empty_country + null_country,
        "country_distribution": dict(country_counts.most_common()),
        "name_length_stats": {
            "min": int(name_arr.min()) if len(name_arr) else 0,
            "mean": float(round(name_arr.mean(), 2)) if len(name_arr) else 0,
            "median": float(round(np.median(name_arr), 2)) if len(name_arr) else 0,
            "max": int(name_arr.max()) if len(name_arr) else 0,
        },
        "addr_length_stats": {
            "min": int(addr_arr.min()) if len(addr_arr) else 0,
            "mean": float(round(addr_arr.mean(), 2)) if len(addr_arr) else 0,
            "median": float(round(np.median(addr_arr), 2)) if len(addr_arr) else 0,
            "max": int(addr_arr.max()) if len(addr_arr) else 0,
        },
        "sample_rows": sample_rows,
        "id_set": id_set
    }


def audit_ground_truth(path: Path, s1_ids: set, s2_ids: set, s3_ids: set) -> Dict[str, Any]:
    print(f"Auditing Ground Truth ({path.name})...")
    gt = load_ground_truth(path)
    total_rows = len(gt)
    
    s1_in_gt = set(gt["source1_entity_id"])
    gt_s1_missing_in_s1 = s1_in_gt - s1_ids
    s1_missing_in_gt = s1_ids - s1_in_gt
    
    match_counts = []
    intra_duplicate_rows = 0
    invalid_prefix_count = 0
    s2_match_count = 0
    s3_match_count = 0
    unresolved_s2_ids = set()
    unresolved_s3_ids = set()
    
    count_freq = Counter()
    
    for _, row in gt.iterrows():
        raw_m = str(row["matched_entity_ids"]).strip()
        if not raw_m or raw_m == "nan":
            match_counts.append(0)
            count_freq[0] += 1
            continue
            
        m_list = [m.strip() for m in raw_m.split(",") if m.strip()]
        match_counts.append(len(m_list))
        count_freq[len(m_list)] += 1
        
        if len(m_list) != len(set(m_list)):
            intra_duplicate_rows += 1
            
        for mid in m_list:
            if mid.startswith("S2-"):
                s2_match_count += 1
                if mid not in s2_ids:
                    unresolved_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                s3_match_count += 1
                if mid not in s3_ids:
                    unresolved_s3_ids.add(mid)
            else:
                invalid_prefix_count += 1

    mc_arr = np.array(match_counts)
    
    zero_matches = int((mc_arr == 0).sum())
    one_match = int((mc_arr == 1).sum())
    two_matches = int((mc_arr == 2).sum())
    three_matches = int((mc_arr == 3).sum())
    four_or_more = int((mc_arr >= 4).sum())
    
    return {
        "total_rows": total_rows,
        "singleton_count (0 matches)": zero_matches,
        "singleton_percentage": float(round(100 * zero_matches / total_rows, 2)),
        "one_match": one_match,
        "two_matches": two_matches,
        "three_matches": three_matches,
        "four_or_more_matches": four_or_more,
        "match_count_frequency": dict(sorted(count_freq.items())),
        "avg_matches": float(round(mc_arr.mean(), 3)),
        "median_matches": float(round(np.median(mc_arr), 3)),
        "max_matches": int(mc_arr.max()),
        "total_s2_links": s2_match_count,
        "total_s3_links": s3_match_count,
        "intra_duplicate_rows": intra_duplicate_rows,
        "invalid_prefix_count": invalid_prefix_count,
        "gt_s1_missing_in_s1_count": len(gt_s1_missing_in_s1),
        "s1_missing_in_gt_count": len(s1_missing_in_gt),
        "unresolved_s2_ids_count": len(unresolved_s2_ids),
        "unresolved_s3_ids_count": len(unresolved_s3_ids)
    }


def main():
    paths = get_default_paths()
    
    # Audit train sources
    t_s1 = audit_source_file(paths["train_source1"], "Train Source 1", "S1-")
    s1_ids = t_s1.pop("id_set")
    
    t_s2 = audit_source_file(paths["train_source2"], "Train Source 2", "S2-")
    s2_ids = t_s2.pop("id_set")
    
    t_s3 = audit_source_file(paths["train_source3"], "Train Source 3", "S3-")
    s3_ids = t_s3.pop("id_set")
    
    gt_audit = audit_ground_truth(paths["train_gt"], s1_ids, s2_ids, s3_ids)
    
    # Free memory of train IDs before test audit
    del s1_ids, s2_ids, s3_ids
    
    # Audit test sources
    te_s1 = audit_source_file(paths["test_source1"], "Test Source 1", "S1-")
    te_s1.pop("id_set")
    
    te_s2 = audit_source_file(paths["test_source2"], "Test Source 2", "S2-")
    te_s2.pop("id_set")
    
    te_s3 = audit_source_file(paths["test_source3"], "Test Source 3", "S3-")
    te_s3.pop("id_set")
    
    eda_summary = {
        "train_source1": t_s1,
        "train_source2": t_s2,
        "train_source3": t_s3,
        "ground_truth": gt_audit,
        "test_source1": te_s1,
        "test_source2": te_s2,
        "test_source3": te_s3,
    }
    
    out_json = Path("experiments/eda_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")
    print(f"EDA JSON report saved to {out_json}")
    
    # Print high level console report
    print("\n" + "="*60)
    print("EDA AUDIT COMPLETE SUMMARY")
    print("="*60)
    print(f"Train Source 1: {t_s1['total_rows']:,} rows, Countries: {t_s1['country_distribution']}")
    print(f"Train Source 2: {t_s2['total_rows']:,} rows, Countries: {t_s2['country_distribution']}")
    print(f"Train Source 3: {t_s3['total_rows']:,} rows, Countries: {t_s3['country_distribution']}")
    print(f"Ground Truth: {gt_audit['total_rows']:,} rows, Singletons (0 matches): {gt_audit['singleton_count (0 matches)']:,} ({gt_audit['singleton_percentage']}%)")
    print(f"GT Matches: avg={gt_audit['avg_matches']}, median={gt_audit['median_matches']}, max={gt_audit['max_matches']}")
    print(f"Total True Links: S2={gt_audit['total_s2_links']:,}, S3={gt_audit['total_s3_links']:,}")
    print(f"Test Source 1: {te_s1['total_rows']:,} rows, Countries: {te_s1['country_distribution']}")
    print(f"Test Source 2: {te_s2['total_rows']:,} rows, Countries: {te_s2['country_distribution']}")
    print(f"Test Source 3: {te_s3['total_rows']:,} rows, Countries: {te_s3['country_distribution']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
