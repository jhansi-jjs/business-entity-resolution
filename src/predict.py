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
import warnings
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
from rapidfuzz import fuzz

from src.data_loader import get_default_paths, load_source_tsv, iter_source_tsv
from src.normalize import normalize_name, normalize_address, extract_postal_code
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
        query_batch_size: int = 10000,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Generate candidates and predict matches for a single country partition with high-throughput vectorized scoring.

        Returns:
            (matching_results_map, candidate_pairs_map)
        """
        logger.info(f"Resolving {country}: {len(s1_df):,} S1 queries vs {len(targets_df):,} candidates...")
        if len(s1_df) == 0:
            return {}, {}

        # 1. Normalization in parallel chunks upfront
        s1_df = s1_df.copy()
        targets_df = targets_df.copy()

        def _norm_chunk(b_names, b_addrs, c_code):
            names_res = [normalize_name(n) for n in b_names]
            addrs_res = [normalize_address(a) for a in b_addrs]
            name_norms = [r[0] for r in names_res]
            name_cores = [r[1] for r in names_res]
            name_asciis = [r[2] for r in names_res]
            addr_norms = [r[0] for r in addrs_res]
            addr_nums = [tuple(r[1]) for r in addrs_res]
            postals = [extract_postal_code(r[1], c_code) for r in addrs_res]
            return name_norms, name_cores, name_asciis, addr_norms, addr_nums, postals

        # Targets normalization
        t_names = targets_df["business_name"].fillna("").astype(str).tolist()
        t_addrs = targets_df["business_address"].fillna("").astype(str).tolist()
        chunk_size = 25000
        t_chunks = [(t_names[i:i+chunk_size], t_addrs[i:i+chunk_size]) for i in range(0, len(t_names), chunk_size)]
        del t_names, t_addrs
        norm_res = joblib.Parallel(n_jobs=8, batch_size=4)(
            joblib.delayed(_norm_chunk)(cn, ca, country) for cn, ca in t_chunks
        )
        del t_chunks

        t_name_norms = [item for r in norm_res for item in r[0]]
        t_name_cores = [item for r in norm_res for item in r[1]]
        t_name_asciis = [item for r in norm_res for item in r[2]]
        t_addr_norms = [item for r in norm_res for item in r[3]]
        t_addr_nums = [item for r in norm_res for item in r[4]]
        t_postals = [item for r in norm_res for item in r[5]]
        del norm_res

        targets_df["name_core"] = t_name_cores
        targets_df["addr_norm"] = t_addr_norms

        # Compact target dictionary: eid -> (norm_n, core_n, ascii_n, norm_a, nums_tuple, postal, is_s3)
        target_dict: Dict[str, Tuple[str, str, str, str, Tuple[str, ...], Optional[str], float]] = {
            eid: (n_norm, n_core, n_ascii, a_norm, a_nums, postal, 1.0 if str(eid).startswith("S3-") else 0.0)
            for eid, n_norm, n_core, n_ascii, a_norm, a_nums, postal in zip(
                targets_df["entity_id"].values,
                t_name_norms,
                t_name_cores,
                t_name_asciis,
                t_addr_norms,
                t_addr_nums,
                t_postals
            )
        }
        del t_name_norms, t_name_cores, t_name_asciis, t_addr_norms, t_addr_nums, t_postals
        if "business_name" in targets_df:
            del targets_df["business_name"]
        if "business_address" in targets_df:
            del targets_df["business_address"]
        gc.collect()

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

        # Free memory from targets_df (blocker already holds its needed arrays)
        del targets_df
        gc.collect()

        # 3. Query normalization and pre-indexing
        q_names = s1_df["business_name"].fillna("").astype(str).tolist()
        q_addrs = s1_df["business_address"].fillna("").astype(str).tolist()
        q_chunks = [(q_names[i:i+chunk_size], q_addrs[i:i+chunk_size]) for i in range(0, len(q_names), chunk_size)]
        del q_names, q_addrs
        q_norm_res = joblib.Parallel(n_jobs=8, batch_size=4)(
            joblib.delayed(_norm_chunk)(cn, ca, country) for cn, ca in q_chunks
        )
        del q_chunks

        q_name_norms = [item for r in q_norm_res for item in r[0]]
        q_name_cores = [item for r in q_norm_res for item in r[1]]
        q_name_asciis = [item for r in q_norm_res for item in r[2]]
        q_addr_norms = [item for r in q_norm_res for item in r[3]]
        q_addr_nums = [item for r in q_norm_res for item in r[4]]
        q_postals = [item for r in q_norm_res for item in r[5]]
        del q_norm_res

        s1_df["name_core"] = q_name_cores
        s1_df["addr_norm"] = q_addr_norms

        # Pre-enrich S1 records: precalculate all token sets, first/last tokens, lengths
        s1_data: Dict[str, Tuple] = {}
        for eid, n_norm, n_core, n_ascii, a_norm, a_nums, postal in zip(
            s1_df["entity_id"].values,
            q_name_norms,
            q_name_cores,
            q_name_asciis,
            q_addr_norms,
            q_addr_nums,
            q_postals
        ):
            s1_tok_list = n_norm.split() if n_norm else []
            s1_data[eid] = (
                n_norm,                                    # 0: name_norm
                n_core,                                    # 1: name_core
                n_ascii,                                   # 2: name_ascii
                set(s1_tok_list),                          # 3: name_toks
                s1_tok_list[0] if s1_tok_list else "",     # 4: name_first_tok
                s1_tok_list[-1] if s1_tok_list else "",    # 5: name_last_tok
                len(n_norm),                               # 6: name_len
                a_norm,                                    # 7: addr_norm
                set(a_norm.split()) if a_norm else set(),  # 8: addr_toks
                set(a_nums),                               # 9: addr_nums_set
                a_nums[0] if a_nums else None,             # 10: addr_first_num
                len(a_norm),                               # 11: addr_len
                postal,                                    # 12: postal
            )
        del q_name_norms, q_name_cores, q_name_asciis, q_addr_norms, q_addr_nums, q_postals
        gc.collect()

        cand_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}
        match_map: Dict[str, List[str]] = {eid: [] for eid in s1_df["entity_id"]}

        # 4. Process S1 queries in batches with high-throughput vectorized scoring
        total_queries = len(s1_df)
        t_partition_start = time.time()
        running_matches = 0

        for start_idx in range(0, total_queries, query_batch_size):
            end_idx = min(start_idx + query_batch_size, total_queries)
            s1_batch = s1_df.iloc[start_idx:end_idx]

            batch_cands = blocker.query(s1_batch, batch_size=500)

            if batch_cands:
                valid_cands = [item for item in batch_cands if item[1] in target_dict]
                n_pairs = len(valid_cands)

                if n_pairs > 0:
                    feats = np.zeros((n_pairs, 31), dtype=np.float32)
                    cand_cache: Dict[str, Tuple] = {}

                    for p_idx, item in enumerate(valid_cands):
                        s1_id = item[0]
                        cid = item[1]
                        n_score = item[3]
                        a_score = item[4]
                        num_m = item[6]

                        cand_map[s1_id].append(cid)

                        q = s1_data[s1_id]
                        c_info = target_dict[cid]

                        c_norm, c_core, c_ascii, c_addr, c_nums, c_postal, c_is_s3 = c_info

                        if cid in cand_cache:
                            c_toks, c_first_tok, c_last_tok, c_a_toks, c_nums_set, c_first_num = cand_cache[cid]
                        else:
                            c_tok_list = c_norm.split() if c_norm else []
                            c_toks = set(c_tok_list)
                            c_first_tok = c_tok_list[0] if c_tok_list else ""
                            c_last_tok = c_tok_list[-1] if c_tok_list else ""
                            c_a_toks = set(c_addr.split()) if c_addr else set()
                            c_nums_set = set(c_nums)
                            c_first_num = c_nums[0] if c_nums else None
                            cand_cache[cid] = (c_toks, c_first_tok, c_last_tok, c_a_toks, c_nums_set, c_first_num)

                        # Name features
                        s1_norm = q[0]
                        s1_core = q[1]
                        s1_ascii = q[2]
                        s1_toks = q[3]
                        s1_first_tok = q[4]
                        s1_last_tok = q[5]
                        s1_len = q[6]

                        if not s1_norm or not c_norm:
                            l_diff = float(abs(s1_len - len(c_norm)))
                            feats[p_idx, 12] = l_diff
                            feats[p_idx, 13] = 1.0
                        else:
                            feats[p_idx, 0] = 1.0 if s1_norm == c_norm else 0.0
                            feats[p_idx, 1] = 1.0 if s1_core and s1_core == c_core else 0.0
                            feats[p_idx, 2] = fuzz.ratio(s1_norm, c_norm) / 100.0
                            feats[p_idx, 3] = (
                                fuzz.ratio(s1_ascii, c_ascii) / 100.0
                                if (s1_ascii and c_ascii)
                                else feats[p_idx, 2]
                            )
                            feats[p_idx, 4] = fuzz.partial_ratio(s1_norm, c_norm) / 100.0
                            feats[p_idx, 5] = fuzz.token_sort_ratio(s1_norm, c_norm) / 100.0
                            feats[p_idx, 6] = fuzz.token_set_ratio(s1_norm, c_norm) / 100.0
                            inter_name = s1_toks.intersection(c_toks)
                            u_name = s1_toks.union(c_toks)
                            feats[p_idx, 7] = len(inter_name) / len(u_name) if u_name else 0.0
                            min_toks = min(len(s1_toks), len(c_toks))
                            feats[p_idx, 8] = len(inter_name) / min_toks if min_toks > 0 else 0.0
                            feats[p_idx, 9] = float(len(inter_name))
                            feats[p_idx, 10] = 1.0 if (s1_first_tok and s1_first_tok == c_first_tok) else 0.0
                            feats[p_idx, 11] = 1.0 if (s1_last_tok and s1_last_tok == c_last_tok) else 0.0
                            l_diff = float(abs(s1_len - len(c_norm)))
                            feats[p_idx, 12] = l_diff
                            feats[p_idx, 13] = l_diff / max(1, max(s1_len, len(c_norm)))

                        # Address features
                        s1_addr = q[7]
                        s1_a_toks = q[8]
                        s1_nums_set = q[9]
                        s1_first_num = q[10]
                        s1_addr_len = q[11]
                        s1_postal = q[12]

                        addr_empty = 1.0 if not c_addr or not c_addr.strip() else 0.0
                        if not addr_empty and s1_addr:
                            feats[p_idx, 14] = 1.0 if s1_addr == c_addr else 0.0
                            feats[p_idx, 15] = fuzz.ratio(s1_addr, c_addr) / 100.0
                            feats[p_idx, 16] = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
                            feats[p_idx, 17] = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
                            feats[p_idx, 18] = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
                            inter_addr = s1_a_toks.intersection(c_a_toks)
                            u_addr = s1_a_toks.union(c_a_toks)
                            feats[p_idx, 19] = len(inter_addr) / len(u_addr) if u_addr else 0.0
                            min_a_toks = min(len(s1_a_toks), len(c_a_toks))
                            feats[p_idx, 20] = len(inter_addr) / min_a_toks if min_a_toks > 0 else 0.0
                            inter_nums = s1_nums_set.intersection(c_nums_set)
                            u_nums = s1_nums_set.union(c_nums_set)
                            feats[p_idx, 21] = float(len(inter_nums))
                            feats[p_idx, 22] = len(inter_nums) / len(u_nums) if u_nums else 0.0
                            feats[p_idx, 23] = (
                                1.0
                                if (s1_first_num and s1_first_num == c_first_num)
                                else 0.0
                            )
                            feats[p_idx, 24] = (
                                1.0
                                if (s1_postal and c_postal and s1_postal == c_postal)
                                else 0.0
                            )
                            feats[p_idx, 25] = float(abs(s1_addr_len - len(c_addr)))

                        # Meta features
                        feats[p_idx, 26] = c_is_s3
                        feats[p_idx, 27] = addr_empty
                        feats[p_idx, 28] = float(n_score)
                        feats[p_idx, 29] = float(a_score)
                        feats[p_idx, 30] = float(num_m)

                    # Model prediction
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        probs = self.model.predict_proba(feats)[:, 1]
                    matched_indices = np.where(probs >= self.threshold)[0]
                    for idx in matched_indices:
                        item = valid_cands[idx]
                        match_map[item[0]].append(item[1])
                        running_matches += 1

                    del feats, cand_cache, probs, matched_indices

            # Periodic progress reporting
            if end_idx % 10000 == 0 or end_idx == total_queries:
                elapsed = time.time() - t_partition_start
                rate = end_idx / max(0.1, elapsed)
                eta_sec = (total_queries - end_idx) / max(0.1, rate)
                logger.info(
                    f"[{country}] {end_idx:,}/{total_queries:,} queries ({end_idx/total_queries*100:.1f}%) "
                    f"in {elapsed:.1f}s ({rate:.1f} q/s, ETA: {eta_sec/60:.1f}m). Matches found: {running_matches:,}"
                )

            del batch_cands
            gc.collect()

        # Clean up partition data
        del target_dict, s1_data, blocker
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
