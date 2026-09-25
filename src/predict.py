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
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Generate candidates and predict matches for a single country partition.

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

        # 2. Blocking
        blocker = CountryBlocker(
            country=country,
            name_top_k=self.name_top_k,
            addr_top_k=self.addr_top_k,
            name_sim_threshold=self.name_sim_threshold,
            addr_sim_threshold=self.addr_sim_threshold,
        )
        blocker.fit(targets_df)
        cands = blocker.query(s1_df, batch_size=2000)
        logger.info(f"Generated {len(cands):,} candidate pairs for {country}.")

        # 3. Pre-enrich records
        s1_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in s1_df.iterrows()}
        cand_dict = {row["entity_id"]: enrich_record_dict(row) for _, row in targets_df.iterrows()}

        # 4. Feature Extraction & Scoring in batches
        cand_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}
        match_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}

        if not cands:
            return match_map, cand_map

        pair_records = []
        pair_meta = []
        for item in cands:
            s1_id = item[0]
            cid = item[1]
            csrc = item[2]
            n_score = item[3]
            a_score = item[4]
            num_m = item[6]

            feats = compute_pair_features(
                s1_dict[s1_id],
                cand_dict[cid],
                name_retrieval_score=n_score,
                addr_retrieval_score=a_score,
                num_blocking_methods=num_m
            )
            pair_records.append(feats)
            pair_meta.append((s1_id, cid))
            cand_map[s1_id].append(cid)

        # Batch prediction
        feat_df = pd.DataFrame(pair_records)[self.feature_names]
        probs = self.model.predict_proba(feat_df)[:, 1]

        # Filter by threshold and group
        for idx, (s1_id, cid) in enumerate(pair_meta):
            if probs[idx] >= self.threshold:
                match_map[s1_id].append(cid)

        # Remove duplicate IDs preserving order
        for s1_id in s1_df["entity_id"]:
            match_map[s1_id] = list(dict.fromkeys(match_map[s1_id]))
            cand_map[s1_id] = list(dict.fromkeys(cand_map[s1_id]))

            # Critical assertion: final matches must be subset of candidates
            assert set(match_map[s1_id]).issubset(set(cand_map[s1_id])), (
                f"Match subset violation for {s1_id}!"
            )

        return match_map, cand_map


def write_submission_files(
    all_matches: Dict[str, List[str]],
    all_candidates: Dict[str, List[str]],
    s1_all_ids: List[str],
    output_dir: Path
) -> Tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv conforming to exact format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    match_file = output_dir / "matching_results.tsv"
    cand_file = output_dir / "candidate_pairs.tsv"

    logger.info(f"Writing {match_file} ({len(s1_all_ids):,} entities)...")
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in s1_all_ids:
            m_list = all_matches.get(s1, [])
            f.write(f"{s1}\t{','.join(m_list)}\n")

    logger.info(f"Writing {cand_file} ({len(s1_all_ids):,} entities)...")
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_all_ids:
            c_list = all_candidates.get(s1, [])
            f.write(f"{s1}\t{','.join(c_list)}\n")

    logger.info("Submission files written successfully.")
    return match_file, cand_file
