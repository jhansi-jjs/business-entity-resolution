"""End-to-end pipeline implementation for Business Entity Resolution Challenge.

Orchestrates all explicit pipeline stages:
[1/8] Data Loading
[2/8] Schema & ID Validation
[3/8] Normalization
[4/8] Candidate Generation (Country-Partitioned Blocking)
[5/8] Feature Generation
[6/8] Matcher Inference & Thresholding
[7/8] Submission Aggregation & Output Writing
[8/8] Organizer Validation Execution
"""

import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import logging
import gc
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd

from src.data_loader import get_default_paths, load_source_tsv, iter_source_tsv
from src.predict import EntityResolver, write_submission_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(
    sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
    run_validator: bool = True
) -> Dict[str, Any]:
    """Execute the complete end-to-end entity resolution pipeline."""
    paths = get_default_paths()
    if output_dir is None:
        if sample_size is not None:
            output_dir = PROJECT_ROOT / "output" / "sample_smoke"
        else:
            output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    print("="*70)

    # Stage 1 & 2: Loading & Validation
    print("\n[1/8] Loading and validating test source datasets...")
    # Load all required S1 entity IDs
    s1_all_ids = []
    with open(paths["test_source1"], "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            if line.strip():
                s1_all_ids.append(line.split("\t", 1)[0].strip())
    print(f"  Total required Test S1 entities: {len(s1_all_ids):,}")

    # Load S1 records (sample or full)
    s1_df = load_source_tsv(paths["test_source1"], "Test Source 1", "S1-", nrows=sample_size)
    print(f"  Loaded {len(s1_df):,} S1 queries for resolution.")

    # Stage 3: Loading matcher and configuration
    print("\n[2/8] Loading trained matcher model and threshold configuration...")
    model_path = PROJECT_ROOT / "models" / "matcher.joblib"
    config_path = PROJECT_ROOT / "models" / "model_config.json"
    resolver = EntityResolver(model_path=model_path, config_path=config_path)

    # Initialize maps for all required S1 entities
    all_matches: Dict[str, List[str]] = {eid: [] for eid in s1_all_ids}
    all_candidates: Dict[str, List[str]] = {eid: [] for eid in s1_all_ids}

    # Stage 4 to 6: By Country Partition (France, US, India)
    countries_to_process = s1_df["country"].unique().tolist()
    print(f"\n[3/8] - [6/8] Processing candidate generation and prediction across countries: {countries_to_process}...")

    total_candidate_pairs = 0
    total_predicted_matches = 0

    for country in countries_to_process:
        s1_country = s1_df[s1_df["country"] == country]
        if len(s1_country) == 0:
            continue

        print(f"\n>>> Processing Country: {country} ({len(s1_country):,} queries) <<<")
        # Load targets for this country
        # In test, load S2 and S3 partition for this country
        s2_chunk = []
        for c in iter_source_tsv(paths["test_source2"], chunksize=250000):
            hit = c[c["country"] == country]
            if len(hit) > 0:
                s2_chunk.append(hit)
            if sample_size and sum(len(h) for h in s2_chunk) >= sample_size * 5:
                break

        s3_chunk = []
        for c in iter_source_tsv(paths["test_source3"], chunksize=250000):
            hit = c[c["country"] == country]
            if len(hit) > 0:
                s3_chunk.append(hit)
            if sample_size and sum(len(h) for h in s3_chunk) >= sample_size * 5:
                break

        targets_country = pd.concat(s2_chunk + s3_chunk, ignore_index=True).drop_duplicates(subset=["entity_id"])
        print(f"  Target pool for {country}: {len(targets_country):,} candidates.")

        c_matches, c_cands = resolver.predict_country_partition(country, s1_country, targets_country)

        # Update global maps
        for s1_id, m_list in c_matches.items():
            all_matches[s1_id] = m_list
            total_predicted_matches += len(m_list)

        for s1_id, c_list in c_cands.items():
            all_candidates[s1_id] = c_list
            total_candidate_pairs += len(c_list)

        del targets_country, s2_chunk, s3_chunk, c_matches, c_cands
        gc.collect()

    # Stage 7: Writing TSV files
    print("\n[7/8] Writing final submission TSV files...")
    match_file, cand_file = write_submission_files(
        all_matches=all_matches,
        all_candidates=all_candidates,
        s1_all_ids=s1_all_ids,
        output_dir=output_dir
    )

    singleton_preds = sum(1 for m in all_matches.values() if len(m) == 0)
    avg_cands = total_candidate_pairs / max(1, len(s1_df))

    print("\nPipeline Execution Statistics:")
    print(f"  Total S1 entities written: {len(s1_all_ids):,}")
    print(f"  Total candidate pairs generated: {total_candidate_pairs:,}")
    print(f"  Average candidates per processed S1: {avg_cands:.2f}")
    print(f"  Total predicted links: {total_predicted_matches:,}")
    print(f"  Singleton (0-match) predictions: {singleton_preds:,} ({(100*singleton_preds/len(s1_all_ids)):.2f}%)")
    print(f"  Decision threshold used: {resolver.threshold}")

    # Stage 8: Organizer Validator
    val_status = "SKIPPED"
    if run_validator:
        print("\n[8/8] Executing organizer submission validator...")
        val_script = PROJECT_ROOT / "utils" / "validate_submission.py"
        test_dir = PROJECT_ROOT / "dataset" / "test"

        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(match_file),
            "--candidate", str(cand_file),
            "--test-dir", str(test_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("\n--- ORGANIZER VALIDATOR OUTPUT ---")
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        print("----------------------------------")
        val_status = "PASS" if res.returncode == 0 else f"FAIL (exit code {res.returncode})"

    return {
        "status": "SUCCESS",
        "validator_status": val_status,
        "total_s1": len(s1_all_ids),
        "candidate_pairs": total_candidate_pairs,
        "predicted_links": total_predicted_matches,
        "singletons": singleton_preds,
        "threshold": resolver.threshold,
        "matching_file": str(match_file),
        "candidate_file": str(cand_file)
    }
