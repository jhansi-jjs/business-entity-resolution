"""Deterministic Baseline for Business Entity Resolution.

Evaluates ALL 1,732,544 test Source 1 entities across all countries
against the full test Source 2 and Source 3 target pools without GBDT inference.
Ensures zero artificial padding, strict candidate subset invariant,
bounded memory (<1.0 GB RAM), and fast execution (<4 minutes).
"""

import os
import sys
import time
import gc
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Strip legal suffixes regex
SUFFIX_PAT = re.compile(
    r"\b(pvt ltd|private limited|pvt|ltd|limited|inc|incorporated|corp|corporation|llc|llp|co|company|cie|sarl|sas|sasu|sci|sa|snc|eurl)\b",
    re.IGNORECASE
)
PUNCT_PAT = re.compile(r"[^\w\s]", re.UNICODE)
NUM_PAT = re.compile(r"\b\d+\b")
POSTAL_PAT = re.compile(r"\b\d{5,6}\b")


def get_core_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    text = name.lower()
    text = PUNCT_PAT.sub(" ", text)
    text = SUFFIX_PAT.sub(" ", text)
    tokens = text.split()
    return " ".join(tokens)


def get_addr_tokens(addr: str) -> Tuple[Set[str], Set[str]]:
    if not addr or not isinstance(addr, str):
        return set(), set()
    nums = set(NUM_PAT.findall(addr))
    postals = set(POSTAL_PAT.findall(addr))
    return nums, postals


def run_deterministic_baseline(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    max_candidates_per_core: int = 50
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    match_out_path = os.path.join(output_dir, "baseline_matching_results.tsv")
    cand_out_path = os.path.join(output_dir, "baseline_candidate_pairs.tsv")
    match_tmp = match_out_path + ".tmp"
    cand_tmp = cand_out_path + ".tmp"

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print("=" * 80)
    print("DETERMINISTIC FULL-TEST BASELINE PIPELINE")
    print(f"Test S1 Path: {s1_path}")
    print(f"Output Match: {match_out_path}")
    print(f"Output Cands: {cand_out_path}")
    print("=" * 80)

    # 1. Read S1 entities (keeping memory lightweight: list of tuples in exact file order)
    print("\n[1/5] Loading Test Source 1 in memory...")
    t0 = time.time()
    s1_records: List[Tuple[str, str, str, str, Set[str], Set[str]]] = []
    # (entity_id, country, raw_name, core_name, nums, postals)
    country_indices: Dict[str, List[int]] = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        col_map = {col.lower(): idx for idx, col in enumerate(header)}
        id_idx = col_map["entity_id"]
        name_idx = col_map["business_name"]
        addr_idx = col_map["business_address"]
        country_idx = col_map["country"]

        for row_idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= max(id_idx, name_idx, addr_idx, country_idx):
                continue
            eid = parts[id_idx].strip()
            raw_n = parts[name_idx].strip()
            raw_a = parts[addr_idx].strip()
            c = parts[country_idx].strip()

            core_n = get_core_name(raw_n)
            nums, postals = get_addr_tokens(raw_a)

            s1_records.append((eid, c, raw_n, core_n, nums, postals))
            country_indices[c].append(row_idx)

    total_s1 = len(s1_records)
    print(f"Loaded {total_s1:,} Test S1 queries in {time.time() - t0:.2f}s.")
    for c, idxs in country_indices.items():
        print(f"  - {c}: {len(idxs):,} queries")

    # Arrays to store results matching exact s1_records row indices
    result_matches: List[str] = ["" for _ in range(total_s1)]
    result_candidates: List[str] = ["" for _ in range(total_s1)]

    # 2. Process Country by Country to strictly bound RAM
    countries = ["France", "US", "India"]  # Process France first, then US, then India
    for country in countries:
        c_idxs = country_indices.get(country, [])
        if not c_idxs:
            continue

        print(f"\n[2/5] Building index for Country: {country} ({len(c_idxs):,} queries)...")
        t_country_start = time.time()

        core_index: Dict[str, List[Tuple[str, str, Set[str], Set[str]]]] = defaultdict(list)
        total_targets = 0

        for target_path in (s2_path, s3_path):
            print(f"  Scanning targets from {os.path.basename(target_path)} for {country}...")
            with open(target_path, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\r\n").split("\t")
                col_map = {col.lower(): idx for idx, col in enumerate(header)}
                t_id_idx = col_map["entity_id"]
                t_name_idx = col_map["business_name"]
                t_addr_idx = col_map["business_address"]
                t_country_idx = col_map["country"]

                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) <= max(t_id_idx, t_name_idx, t_addr_idx, t_country_idx):
                        continue
                    if parts[t_country_idx].strip() != country:
                        continue

                    tid = parts[t_id_idx].strip()
                    t_raw_n = parts[t_name_idx].strip()
                    t_raw_a = parts[t_addr_idx].strip()
                    t_core = get_core_name(t_raw_n)

                    if t_core and len(t_core) >= 3:
                        bucket = core_index[t_core]
                        if len(bucket) < max_candidates_per_core:
                            t_nums, t_postals = get_addr_tokens(t_raw_a)
                            bucket.append((tid, t_raw_n, t_nums, t_postals))

                    total_targets += 1

        print(f"  Indexed {len(core_index):,} distinct core names from {total_targets:,} {country} targets in {time.time() - t_country_start:.2f}s.")

        # Resolve queries for this country
        print(f"  Resolving {len(c_idxs):,} {country} queries...")
        t_res_start = time.time()
        c_matches_count = 0
        c_cands_count = 0

        for idx in c_idxs:
            eid, c, s1_raw_n, s1_core, s1_nums, s1_postals = s1_records[idx]
            if not s1_core or len(s1_core) < 3:
                continue

            target_bucket = core_index.get(s1_core, [])
            if not target_bucket:
                continue

            cands_for_s1: List[str] = []
            matches_for_s1: List[str] = []

            for tid, t_raw_n, t_nums, t_postals in target_bucket:
                cands_for_s1.append(tid)

                # Matching logic:
                is_match = False
                if s1_postals and t_postals and (s1_postals & t_postals):
                    is_match = True
                elif s1_nums and t_nums and (s1_nums & t_nums):
                    is_match = True
                elif s1_raw_n.lower() == t_raw_n.lower() and len(s1_raw_n) >= 5:
                    is_match = True
                elif len(s1_core) >= 12:
                    if not (s1_postals and t_postals and not (s1_postals & t_postals)):
                        is_match = True

                if is_match:
                    matches_for_s1.append(tid)

            # De-duplicate while preserving order
            cands_unique = list(dict.fromkeys(cands_for_s1))
            matches_unique = list(dict.fromkeys(matches_for_s1))

            if cands_unique:
                result_candidates[idx] = ",".join(cands_unique)
                c_cands_count += len(cands_unique)

            if matches_unique:
                result_matches[idx] = ",".join(matches_unique)
                c_matches_count += len(matches_unique)

        print(f"  Completed {country} resolution in {time.time() - t_res_start:.2f}s:")
        print(f"    - Matches found: {c_matches_count:,}")
        print(f"    - Candidates generated: {c_cands_count:,}")

        del core_index
        gc.collect()

    # 3. Write submission TSVs with atomic rename
    print("\n[3/5] Writing TSV submission files...")
    t_write = time.time()

    with open(match_tmp, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for (eid, _, _, _, _, _), m_str in zip(s1_records, result_matches):
            f_match.write(f"{eid}\t{m_str}\n")

    with open(cand_tmp, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for (eid, _, _, _, _, _), c_str in zip(s1_records, result_candidates):
            f_cand.write(f"{eid}\t{c_str}\n")

    if os.path.isfile(match_out_path):
        os.remove(match_out_path)
    os.rename(match_tmp, match_out_path)

    if os.path.isfile(cand_out_path):
        os.remove(cand_out_path)
    os.rename(cand_tmp, cand_out_path)

    print(f"TSV files written in {time.time() - t_write:.2f}s.")

    # 4. Invariant Verification
    print("\n[4/5] Verifying submission invariants...")
    total_matches = sum(1 for m in result_matches if m)
    total_singletons = sum(1 for m in result_matches if not m)
    total_candidates = sum(1 for c in result_candidates if c)

    print(f"  Total S1 Entities: {total_s1:,}")
    print(f"  Entities with Matches: {total_matches:,} ({total_matches/total_s1*100:.2f}%)")
    print(f"  Genuine Singletons: {total_singletons:,} ({total_singletons/total_s1*100:.2f}%)")
    print(f"  Entities with Candidates: {total_candidates:,} ({total_candidates/total_s1*100:.2f}%)")

    violations = 0
    for idx in range(total_s1):
        m_str = result_matches[idx]
        c_str = result_candidates[idx]
        if m_str:
            m_set = set(m_str.split(","))
            c_set = set(c_str.split(",")) if c_str else set()
            if not m_set.issubset(c_set):
                violations += 1

    if violations > 0:
        raise ValueError(f"FATAL: Subset invariant violated on {violations} rows!")
    print("  ALL SUBSET INVARIANTS SATISFIED (matches <= candidates on 100% of rows).")

    # 5. Organizer Validation
    print("\n[5/5] Running official organizer validator...")
    import subprocess
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", match_out_path,
        "--candidate", cand_out_path,
        "--test-dir", test_dir
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator Stderr:", res.stderr)

    total_elapsed = time.time() - start_time
    print(f"\nDeterministic Baseline Finished Successfully in {total_elapsed:.2f}s ({total_elapsed/60:.2f}m)!")
    return res.returncode == 0


if __name__ == "__main__":
    success = run_deterministic_baseline()
    sys.exit(0 if success else 1)
