"""Prediction module for Business Entity Resolution Challenge.

Loads test sets, performs country-partitioned blocking, builds features,
scores candidates with the trained matcher, applies the validated threshold,
and writes matching_results.tsv and candidate_pairs.tsv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import logging
import gc
import time
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib

from src.data_loader import get_default_paths, load_source_tsv, iter_source_tsv
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker
from src.features import FEATURE_NAMES, compute_pair_features, enrich_record_dict

logger = logging.getLogger(__name__)


class EntityResolver:
    """End-to-end predictor for Business Entity Resolution."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        name_top_k: int = 25,
        addr_top_k: int = 8,
        name_sim_threshold: float = 0.10,
        addr_sim_threshold: float = 0.18,
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at {model_path}!")
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found at {config_path}!")

        logger.info(f"Loading trained matcher from {model_path}...")
        self.model = joblib.load(model_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.threshold = float(self.config.get("threshold", 0.75))
        self.feature_names = self.config.get("feature_names", FEATURE_NAMES)
        logger.info(f"Loaded {self.config.get('best_model', 'Model')} with threshold={self.threshold}")

        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.name_sim_threshold = name_sim_threshold
        self.addr_sim_threshold = addr_sim_threshold

    def predict_country_partition(
        self,
        country: str,
        s1_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        query_batch_size: int = 5000,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Generate candidates and predict matches for a single country partition in bounded batches.

        Returns:
            (matching_results_map, candidate_pairs_map)
        """
        logger.info(f"Resolving {country}: {len(s1_df):,} S1 queries vs {len(targets_df):,} candidates...")
        if len(s1_df) == 0:
            return {}, {}

        # 1. Normalization
        s1_df = s1_df.copy()
        targets_df = targets_df.copy()

        s1_df["name_core"] = [normalize_name(n)[1] for n in s1_df["business_name"]]
        s1_df["addr_norm"] = [normalize_address(a)[0] for a in s1_df["business_address"]]
        targets_df["name_core"] = [normalize_name(n)[1] for n in targets_df["business_name"]]
        targets_df["addr_norm"] = [normalize_address(a)[0] for a in targets_df["business_address"]]

        # Lightweight target lookup tuple (bname, baddr, country) to avoid multi-GB memory overhead
        target_lookup = {
            eid: (n, a, c) for eid, n, a, c in zip(
                targets_df["entity_id"].values,
                targets_df["business_name"].fillna("").values,
                targets_df["business_address"].fillna("").values,
                targets_df["country"].fillna("").values
            )
        }

        # 2. Fit Blocker once on target pool
        blocker = CountryBlocker(
            country=country,
            name_top_k=self.name_top_k,
            addr_top_k=self.addr_top_k,
            name_sim_threshold=self.name_sim_threshold,
            addr_sim_threshold=self.addr_sim_threshold,
            backend="inverted"
        )
        blocker.fit(targets_df)

        cand_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}
        match_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}

        # 3. Process S1 queries in strictly bounded batches
        total_queries = len(s1_df)
        t_partition_start = time.time()
        running_matches = 0

        for start_idx in range(0, total_queries, query_batch_size):
            end_idx = min(start_idx + query_batch_size, total_queries)
            s1_batch = s1_df.iloc[start_idx:end_idx]

            s1_batch_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in s1_batch.iterrows()}
            batch_cands = blocker.query(s1_batch, batch_size=500)

            if batch_cands:
                needed_cids = {item[1] for item in batch_cands if item[1] in target_lookup}
                batch_cand_dict = {
                    cid: enrich_record_dict({
                        "entity_id": cid,
                        "business_name": target_lookup[cid][0],
                        "business_address": target_lookup[cid][1],
                        "country": target_lookup[cid][2]
                    })
                    for cid in needed_cids
                }

                pair_records = []
                pair_meta = []
                for item in batch_cands:
                    s1_id = item[0]
                    cid = item[1]
                    csrc = item[2]
                    n_score = item[3]
                    a_score = item[4]
                    num_m = item[6]

                    if cid not in batch_cand_dict:
                        continue

                    feats = compute_pair_features(
                        s1_batch_dict[s1_id],
                        batch_cand_dict[cid],
                        name_retrieval_score=n_score,
                        addr_retrieval_score=a_score,
                        num_blocking_methods=num_m
                    )
                    pair_records.append(feats)
                    pair_meta.append((s1_id, cid))
                    cand_map[s1_id].append(cid)

                if pair_records:
                    feat_df = pd.DataFrame(pair_records)[self.feature_names]
                    probs = self.model.predict_proba(feat_df)[:, 1]

                    for idx, (s1_id, cid) in enumerate(pair_meta):
                        if probs[idx] >= self.threshold:
                            match_map[s1_id].append(cid)
                            running_matches += 1

                del pair_records, pair_meta, batch_cand_dict, feat_df, probs

            # Periodic progress reporting
            if end_idx % 20000 == 0 or end_idx == total_queries:
                elapsed = time.time() - t_partition_start
                rate = end_idx / max(0.1, elapsed)
                eta_sec = (total_queries - end_idx) / max(0.1, rate)
                logger.info(
                    f"[{country}] {end_idx:,}/{total_queries:,} queries ({end_idx/total_queries*100:.1f}%) "
                    f"in {elapsed:.1f}s ({rate:.1f} q/s, ETA: {eta_sec/60:.1f}m). Matches found: {running_matches:,}"
                )

            del batch_cands, s1_batch_dict
            gc.collect()

        # Clean up target lookup
        del target_lookup
        gc.collect()

        # Deduplicate and verify invariant
        for s1_id in s1_df["entity_id"]:
            match_map[s1_id] = list(dict.fromkeys(match_map[s1_id]))
            cand_map[s1_id] = list(dict.fromkeys(cand_map[s1_id]))

            # Critical assertion: final matches must be subset of candidates
            assert set(match_map[s1_id]).issubset(set(cand_map[s1_id])), (
                f"Match subset violation for {s1_id}!"
            )

        logger.info(f"Resolved {country}: {sum(len(m) for m in match_map.values()):,} matches predicted across {len(s1_df):,} queries.")
        return match_map, cand_map


def write_submission_files(
    all_matches: Dict[str, List[str]],
    all_candidates: Dict[str, List[str]],
    s1_all_ids: List[str],
    output_dir: Path
) -> Tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv atomically conforming to exact format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    match_file = output_dir / "matching_results.tsv"
    cand_file = output_dir / "candidate_pairs.tsv"
    match_tmp = output_dir / "matching_results.tsv.tmp"
    cand_tmp = output_dir / "candidate_pairs.tsv.tmp"

    logger.info(f"Writing {match_tmp} ({len(s1_all_ids):,} entities)...")
    written_match_count = 0
    with open(match_tmp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_all_ids:
            m_list = all_matches.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")
            written_match_count += 1

    assert written_match_count == len(s1_all_ids), (
        f"Mismatch in written match rows: {written_match_count} vs {len(s1_all_ids)}"
    )

    logger.info(f"Writing {cand_tmp} ({len(s1_all_ids):,} entities)...")
    written_cand_count = 0
    with open(cand_tmp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_all_ids:
            c_list = all_candidates.get(s1, [])
            f.write(f"{s1}\t{','.join(c_list)}\n")
            written_cand_count += 1

    assert written_cand_count == len(s1_all_ids), (
        f"Mismatch in written candidate rows: {written_cand_count} vs {len(s1_all_ids)}"
    )

    # Atomic promotion
    if match_file.exists():
        match_file.unlink()
    match_tmp.rename(match_file)

    if cand_file.exists():
        cand_file.unlink()
    cand_tmp.rename(cand_file)

    logger.info("Submission files written and atomically promoted successfully.")
    return match_file, cand_file
