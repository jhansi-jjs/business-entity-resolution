"""Candidate generation and blocking module for Business Entity Resolution Challenge.

Performs scalable, country-partitioned multi-strategy candidate retrieval using:
1. Strategy A: Character n-gram TF-IDF on normalized business names
2. Strategy B: Character n-gram TF-IDF on normalized business addresses
3. Strategy C: Exact name core inverted index matching
4. Multi-strategy union with diagnostics
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import logging

from src.normalize import normalize_name, normalize_address

logger = logging.getLogger(__name__)


class CountryBlocker:
    """Handles multi-strategy indexing and candidate retrieval for one country partition."""

    def __init__(
        self,
        country: str,
        name_ngram_range: Tuple[int, int] = (3, 5),
        addr_ngram_range: Tuple[int, int] = (3, 5),
        name_top_k: int = 25,
        addr_top_k: int = 10,
        name_sim_threshold: float = 0.12,
        addr_sim_threshold: float = 0.20,
    ):
        self.country = country
        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.name_sim_threshold = name_sim_threshold
        self.addr_sim_threshold = addr_sim_threshold

        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=name_ngram_range,
            min_df=2,
            dtype=np.float32,
            sublinear_tf=True
        )
        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=addr_ngram_range,
            min_df=2,
            dtype=np.float32,
            sublinear_tf=True
        )

        self.target_matrix_name: Optional[csr_matrix] = None
        self.target_matrix_addr: Optional[csr_matrix] = None
        self.target_ids: List[str] = []
        self.target_sources: List[str] = []
        self.target_exact_map: Dict[str, List[int]] = {}

    def fit(self, target_df: pd.DataFrame) -> "CountryBlocker":
        """Index target (Source 2 + Source 3) records for this country."""
        self.target_ids = target_df["entity_id"].tolist()
        self.target_sources = ["S2" if eid.startswith("S2-") else "S3" for eid in self.target_ids]

        cores = target_df["name_core"].fillna("").astype(str).tolist()
        addrs = target_df["addr_norm"].fillna("").astype(str).tolist()

        # Fit TF-IDF on names
        self.target_matrix_name = self.name_vectorizer.fit_transform(cores)

        # Fit TF-IDF on addresses
        self.target_matrix_addr = self.addr_vectorizer.fit_transform(addrs)

        # Build exact core inverted index
        self.target_exact_map = {}
        for idx, core in enumerate(cores):
            c_str = core.strip()
            if c_str:
                self.target_exact_map.setdefault(c_str, []).append(idx)

        return self

    def query(
        self,
        query_df: pd.DataFrame,
        batch_size: int = 2000
    ) -> List[Tuple[str, str, str, float, float, str, int]]:
        """Retrieve candidates for a batch of query (Source 1) records.

        Returns list of tuples:
            (s1_id, candidate_id, candidate_source, name_score, addr_score, blocking_reasons, num_methods)
        """
        if self.target_matrix_name is None:
            raise ValueError(f"CountryBlocker for {self.country} has not been fitted!")

        results = []
        query_ids = query_df["entity_id"].tolist()
        cores = query_df["name_core"].fillna("").astype(str).tolist()
        addrs = query_df["addr_norm"].fillna("").astype(str).tolist()

        num_queries = len(query_df)
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            b_ids = query_ids[start_idx:end_idx]
            b_cores = cores[start_idx:end_idx]
            b_addrs = addrs[start_idx:end_idx]

            # 1. Name TF-IDF similarity
            q_mat_name = self.name_vectorizer.transform(b_cores)
            sim_name = q_mat_name.dot(self.target_matrix_name.T)

            # 2. Address TF-IDF similarity
            q_mat_addr = self.addr_vectorizer.transform(b_addrs)
            sim_addr = q_mat_addr.dot(self.target_matrix_addr.T)

            for i, s1_id in enumerate(b_ids):
                # candidate_id -> dict with details
                candidate_dict = {}

                # Strategy C: Exact Core Match
                q_core = b_cores[i].strip()
                if q_core and q_core in self.target_exact_map:
                    for t_idx in self.target_exact_map[q_core][:self.name_top_k]:
                        cid = self.target_ids[t_idx]
                        csrc = self.target_sources[t_idx]
                        candidate_dict[cid] = {
                            "s1_id": s1_id,
                            "candidate_id": cid,
                            "candidate_source": csrc,
                            "name_score": 1.0,
                            "addr_score": 0.0,
                            "reasons": {"exact_core"},
                        }

                # Strategy A: Name TF-IDF Top-K
                row_n = sim_name.getrow(i)
                if row_n.nnz > 0:
                    d_n, idx_n = row_n.data, row_n.indices
                    top_k_n = min(self.name_top_k, len(d_n))
                    top_arg = np.argpartition(d_n, -top_k_n)[-top_k_n:]
                    top_arg = top_arg[np.argsort(-d_n[top_arg])]

                    for a in top_arg:
                        score = float(d_n[a])
                        if score < self.name_sim_threshold:
                            break
                        t_idx = int(idx_n[a])
                        cid = self.target_ids[t_idx]
                        csrc = self.target_sources[t_idx]

                        if cid not in candidate_dict:
                            candidate_dict[cid] = {
                                "s1_id": s1_id,
                                "candidate_id": cid,
                                "candidate_source": csrc,
                                "name_score": score,
                                "addr_score": 0.0,
                                "reasons": {"name_tfidf"},
                            }
                        else:
                            candidate_dict[cid]["name_score"] = max(candidate_dict[cid]["name_score"], score)
                            candidate_dict[cid]["reasons"].add("name_tfidf")

                # Strategy B: Address TF-IDF Top-K (for candidates that also have non-trivial name overlap >= 0.08)
                row_a = sim_addr.getrow(i)
                if row_a.nnz > 0:
                    d_a, idx_a = row_a.data, row_a.indices
                    top_k_a = min(self.addr_top_k, len(d_a))
                    top_arg_a = np.argpartition(d_a, -top_k_a)[-top_k_a:]
                    top_arg_a = top_arg_a[np.argsort(-d_a[top_arg_a])]

                    for a in top_arg_a:
                        score = float(d_a[a])
                        if score < self.addr_sim_threshold:
                            break
                        t_idx = int(idx_a[a])
                        cid = self.target_ids[t_idx]
                        csrc = self.target_sources[t_idx]

                        if cid not in candidate_dict:
                            candidate_dict[cid] = {
                                "s1_id": s1_id,
                                "candidate_id": cid,
                                "candidate_source": csrc,
                                "name_score": 0.0,
                                "addr_score": score,
                                "reasons": {"addr_tfidf"},
                            }
                        else:
                            candidate_dict[cid]["addr_score"] = max(candidate_dict[cid]["addr_score"], score)
                            candidate_dict[cid]["reasons"].add("addr_tfidf")

                for c_entry in candidate_dict.values():
                    results.append((
                        c_entry["s1_id"],
                        c_entry["candidate_id"],
                        c_entry["candidate_source"],
                        round(c_entry["name_score"], 4),
                        round(c_entry["addr_score"], 4),
                        ";".join(sorted(c_entry["reasons"])),
                        len(c_entry["reasons"])
                    ))

        return results


def evaluate_blocking(
    candidates_list: List[Tuple],
    ground_truth_map: Dict[str, Set[str]],
    s1_ids: List[str]
) -> Dict[str, float]:
    """Calculate true-link candidate recall, full-recall rate, and reduction ratio."""
    candidate_map: Dict[str, Set[str]] = {}
    for item in candidates_list:
        s1 = item[0]
        cid = item[1]
        candidate_map.setdefault(s1, set()).add(cid)

    total_true_links = 0
    recalled_true_links = 0
    s1_full_recall_count = 0
    s1_at_least_one_hit = 0
    s1_with_matches = 0

    candidate_counts = []

    for s1 in s1_ids:
        c_set = candidate_map.get(s1, set())
        candidate_counts.append(len(c_set))
        true_set = ground_truth_map.get(s1, set())

        if true_set:
            s1_with_matches += 1
            n_true = len(true_set)
            total_true_links += n_true
            hits = len(true_set & c_set)
            recalled_true_links += hits

            if hits == n_true:
                s1_full_recall_count += 1
            if hits > 0:
                s1_at_least_one_hit += 1

    candidate_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    link_recall = recalled_true_links / total_true_links if total_true_links > 0 else 1.0
    full_s1_recall = s1_full_recall_count / s1_with_matches if s1_with_matches > 0 else 1.0
    at_least_one_rate = s1_at_least_one_hit / s1_with_matches if s1_with_matches > 0 else 1.0

    return {
        "total_queries": len(s1_ids),
        "total_true_links": total_true_links,
        "recalled_true_links": recalled_true_links,
        "true_link_recall": round(link_recall, 4),
        "s1_full_match_rate": round(full_s1_recall, 4),
        "s1_at_least_one_hit_rate": round(at_least_one_rate, 4),
        "avg_candidates_per_s1": round(float(candidate_arr.mean()), 2),
        "median_candidates_per_s1": round(float(np.median(candidate_arr)), 2),
        "max_candidates_per_s1": int(candidate_arr.max()) if len(candidate_arr) else 0,
        "total_candidate_pairs": len(candidates_list)
    }
