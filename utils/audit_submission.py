"""Audit and spot-check test predictions for Business Entity Resolution.

Inspects matching_results.tsv, candidate_pairs.tsv, computes distribution metrics,
and pulls random real test matches from test_source1, test_source2, and test_source3.
Usage:
    python utils/audit_submission.py [--samples 15]
"""

import sys
from pathlib import Path
import random
from collections import Counter
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import get_default_paths


def main():
    parser = argparse.ArgumentParser(description="Audit and spot-check submission files")
    parser.add_argument("--samples", type=int, default=15, help="Number of real test matches to inspect")
    parser.add_argument("--matching", type=Path, default=PROJECT_ROOT / "output" / "matching_results.tsv")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "output" / "candidate_pairs.tsv")
    args = parser.parse_args()

    paths = get_default_paths()
    matching_file = args.matching
    candidate_file = args.candidate

    print("=" * 70)
    print("SUBMISSION AUDIT & SPOT-CHECK VERIFICATION")
    print("=" * 70)

    # 1. Row counts & File Integrity
    print("\n[1/3] Checking File Integrity and Line Counts...")
    match_lines = 0
    empty_matches = 0
    match_map = {}
    with open(matching_file, "r", encoding="utf-8") as f:
        header = next(f).strip()
        assert header == "source1_entity_id\tmatched_entity_ids", f"Invalid match header: {header}"
        for line in f:
            match_lines += 1
            parts = line.strip().split("\t", 1)
            s1 = parts[0]
            m = parts[1] if len(parts) > 1 else ""
            if not m:
                empty_matches += 1
                match_map[s1] = []
            else:
                match_map[s1] = [x.strip() for x in m.split(",") if x.strip()]

    cand_lines = 0
    empty_cands = 0
    total_cand_pairs = 0
    subset_violations = 0
    with open(candidate_file, "r", encoding="utf-8") as f:
        header = next(f).strip()
        assert header == "source1_entity_id\tcandidate_entity_ids", f"Invalid cand header: {header}"
        for line in f:
            cand_lines += 1
            parts = line.strip().split("\t", 1)
            s1 = parts[0]
            c = parts[1] if len(parts) > 1 else ""
            if not c:
                empty_cands += 1
                cand_set = set()
            else:
                c_list = [x.strip() for x in c.split(",") if x.strip()]
                cand_set = set(c_list)
                total_cand_pairs += len(c_list)

            # Verify subset invariant
            if s1 in match_map:
                m_set = set(match_map[s1])
                if not m_set.issubset(cand_set):
                    subset_violations += 1

    print(f"  matching_results.tsv rows: {match_lines:,}")
    print(f"  candidate_pairs.tsv rows:  {cand_lines:,}")
    print(f"  Singletons (0 matches):    {empty_matches:,} ({(100 * empty_matches / match_lines):.2f}%)")
    print(f"  Total candidate pairs:     {total_cand_pairs:,} ({(total_cand_pairs / cand_lines):.2f} / query)")
    print(f"  Subset Violations:         {subset_violations} (MUST BE 0)")

    assert match_lines == 1732544, f"Mismatch in match rows: {match_lines}"
    assert cand_lines == 1732544, f"Mismatch in cand rows: {cand_lines}"
    assert subset_violations == 0, f"Found {subset_violations} subset violations!"

    # 2. Cluster Size Distribution
    print("\n[2/3] Analyzing Match Cluster Size Distribution...")
    cluster_sizes = [len(m) for m in match_map.values() if len(m) > 0]
    size_counts = Counter(cluster_sizes)
    print(f"  Total matched entities:    {len(cluster_sizes):,}")
    print(f"  Total predicted links:     {sum(cluster_sizes):,}")
    print(f"  Average links per matched: {sum(cluster_sizes) / len(cluster_sizes):.2f}")
    print("  Cluster size breakdown:")
    for size in sorted(size_counts.keys())[:10]:
        print(f"    Size {size:>2}: {size_counts[size]:>8,} entities ({(100 * size_counts[size] / len(cluster_sizes)):.1f}%)")
    max_size = max(cluster_sizes) if cluster_sizes else 0
    print(f"  Max cluster size:          {max_size}")

    # 3. Spot-check real test entity matches with full names & addresses
    print(f"\n[3/3] Pulling {args.samples} Real Test Entities for Human Visual Audit...")
    random.seed(42)
    sample_eids = random.sample([eid for eid, m in match_map.items() if len(m) >= 2], args.samples)

    s1_lookup = {}
    with open(paths["test_source1"], "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in sample_eids:
                s1_lookup[parts[0]] = (parts[1], parts[2], parts[3])

    needed_target_ids = set()
    for eid in sample_eids:
        needed_target_ids.update(match_map[eid])

    target_lookup = {}
    for src_path in [paths["test_source2"], paths["test_source3"]]:
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if parts[0] in needed_target_ids:
                    target_lookup[parts[0]] = (parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")

    print(f"\n--- {args.samples} REAL TEST MATCHES AUDIT ---")
    for idx, s1_id in enumerate(sample_eids, 1):
        q_name, q_addr, q_country = s1_lookup.get(s1_id, ("?", "?", "?"))
        matched_ids = match_map[s1_id]
        print(f"\n[Case {idx:2d}] Country: {q_country} | Query S1 ID: {s1_id}")
        print(f"  QUERY  NAME: {q_name}")
        print(f"  QUERY  ADDR: {q_addr}")
        for mid in matched_ids:
            t_name, t_addr = target_lookup.get(mid, ("?", "?"))
            print(f"    --> MATCH ({mid}): '{t_name}' | '{t_addr}'")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
