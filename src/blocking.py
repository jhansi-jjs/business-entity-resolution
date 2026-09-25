"""Candidate generation and blocking module for Business Entity Resolution Challenge.

Performs scalable, country-partitioned multi-strategy candidate retrieval using:
1. Fast Inverted Index Backend: Exact core + prefix + token + postal code blocking keys with RapidFuzz scoring
2. Classical TF-IDF Backend: Character n-gram TF-IDF on names and addresses
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Set, Tuple, Optional, Union, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from rapidfuzz import fuzz
import logging

from src.normalize import normalize_name, normalize_address, extract_postal_code

logger = logging.getLogger(__name__)

# Shared canonical blocking configuration
DEFAULT_BLOCKING_CONFIG = {
    "name_ngram_range": (3, 5),
    "addr_ngram_range": (3, 5),
    "name_top_k": 25,
    "addr_top_k": 8,
    "name_sim_threshold": 0.10,
    "addr_sim_threshold": 0.18,
    "min_df": 2,
    "backend": "inverted"
}

GENERIC_STOPWORDS = {
    "pvt", "ltd", "inc", "llc", "corp", "corporation", "private", "limited",
    "the", "and", "co", "company", "group", "enterprises", "solutions",
    "services", "technologies", "tech", "trading", "retail", "international",
    "sa", "sarl", "sas", "gmbh", "holding", "holdings"
}


def extract_blocking_keys(name_core: str, addr_norm: str, country: str) -> Set[Tuple[str, str]]:
    """Extract multi-strategy indexing keys for candidate retrieval."""
    keys = set()
    if not name_core:
        return keys

    words = [w for w in name_core.split() if w not in GENERIC_STOPWORDS and len(w) >= 2]

    # 1. Exact core
    keys.add(("core", name_core))

    # 2. First significant word
    if words:
        keys.add(("w1", words[0]))

    # 3. First two words
    if len(words) >= 2:
        keys.add(("w2", f"{words[0]}_{words[1]}"))

    # 4. Prefix 4
    if len(name_core) >= 4:
        keys.add(("pref4", name_core[:4]))

    # 5. Rare words
    for w in words[1:4]:
        if len(w) >= 3:
            keys.add(("word", w))

    # 6. Postal code + first word
    pin = extract_postal_code(addr_norm, country)
    if pin and words:
        keys.add(("pin_w1", f"{pin}_{words[0][:4]}"))

    return keys


class CountryBlocker:
    """Handles multi-strategy indexing and candidate retrieval for one country partition."""

    def __init__(
        self,
        country: str,
        name_ngram_range: Tuple[int, int] = (3, 5),
        addr_ngram_range: Tuple[int, int] = (3, 5),
        name_top_k: int = 25,
        addr_top_k: int = 8,
        name_sim_threshold: float = 0.10,
        addr_sim_threshold: float = 0.18,
        min_df: int = 2,
        backend: str = "inverted",
        max_postings: int = 1500,
    ):
        self.country = country
        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.name_sim_threshold = name_sim_threshold
        self.addr_sim_threshold = addr_sim_threshold
        self.min_df = min_df
        self.backend = backend
        self.max_postings = max_postings

        # Inverted index structures
        self.index = defaultdict(list)
        self.key_counts = defaultdict(int)

        # TF-IDF structures
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=name_ngram_range,
            min_df=min_df,
            dtype=np.float32,
            sublinear_tf=True
        )
        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=addr_ngram_range,
            min_df=min_df,
            dtype=np.float32,
            sublinear_tf=True
        )

        self.target_matrix_name: Optional[csr_matrix] = None
        self.target_matrix_addr: Optional[csr_matrix] = None
        self.target_ids: List[str] = []
        self.target_names: List[str] = []
        self.target_sources: List[str] = []
        self.target_addrs: List[str] = []
        self.target_exact_map: Dict[str, List[int]] = {}
        self.is_empty_partition = False

    def fit(self, target_df: pd.DataFrame) -> "CountryBlocker":
        """Index target (Source 2 + Source 3) records for this country."""
        if len(target_df) == 0:
            logger.warning(f"Target partition for country '{self.country}' is empty.")
            self.is_empty_partition = True
            return self

        eids = target_df["entity_id"].values
        cores = target_df["name_core"].fillna("").astype(str).values
        addrs = target_df["addr_norm"].fillna("").astype(str).values

        self.target_ids = list(eids)
        self.target_names = list(cores)
        self.target_addrs = list(addrs)
        self.target_sources = ["S2" if str(eid).startswith("S2-") else "S3" for eid in eids]

        if self.backend == "inverted":
            self.index = defaultdict(list)
            self.key_counts = defaultdict(int)

            for idx, (core, addr) in enumerate(zip(cores, addrs)):
                keys = extract_blocking_keys(core, addr, self.country)
                for k in keys:
                    if self.key_counts[k] < self.max_postings:
                        self.index[k].append(idx)
                        self.key_counts[k] += 1
            return self

        # Classical TF-IDF path
        try:
            self.target_matrix_name = self.name_vectorizer.fit_transform(cores)
        except ValueError:
            self.name_vectorizer.set_params(min_df=1)
            self.target_matrix_name = self.name_vectorizer.fit_transform(cores)

        try:
            self.target_matrix_addr = self.addr_vectorizer.fit_transform(addrs)
        except ValueError:
            self.addr_vectorizer.set_params(min_df=1)
            self.target_matrix_addr = self.addr_vectorizer.fit_transform(addrs)

        self.target_exact_map = {}
        for idx, core in enumerate(cores):
            c_str = core.strip()
            if c_str:
                self.target_exact_map.setdefault(c_str, []).append(idx)

        return self

    def query(
        self,
        query_df: pd.DataFrame,
        batch_size: int = 1000
    ) -> List[Tuple[str, str, str, float, float, str, int]]:
        """Retrieve candidates for a batch of query (Source 1) records."""
        if self.is_empty_partition or len(query_df) == 0:
            return []

        if self.backend == "inverted":
            return self._query_inverted(query_df)
        else:
            return self._query_tfidf(query_df, batch_size=batch_size)

    def _query_inverted(
        self,
        query_df: pd.DataFrame
    ) -> List[Tuple[str, str, str, float, float, str, int]]:
        """Fast inverted-index retrieval with candidate ranking."""
        results = []
        q_ids = query_df["entity_id"].values
        q_names = query_df["name_core"].fillna("").astype(str).values
        q_addrs = query_df["addr_norm"].fillna("").astype(str).values

        for s1_id, q_n, q_a in zip(q_ids, q_names, q_addrs):
            q_keys = extract_blocking_keys(q_n, q_a, self.country)
            cand_indices = set()
            for k in q_keys:
                if k in self.index:
                    cand_indices.update(self.index[k])

            if not cand_indices:
                continue

            cand_scores = []
            for c_idx in cand_indices:
                t_n = self.target_names[c_idx]
                t_a = self.target_addrs[c_idx]

                n_sim = fuzz.ratio(q_n, t_n) / 100.0
                a_sim = fuzz.ratio(q_a, t_a) / 100.0 if (q_a and t_a) else 0.0
                combined = 0.70 * n_sim + 0.30 * a_sim if (q_a and t_a) else n_sim

                # Only consider candidates with minimal plausibility
                if combined >= 0.20 or n_sim >= 0.30:
                    cand_scores.append((combined, n_sim, a_sim, c_idx))

            cand_scores.sort(key=lambda x: x[0], reverse=True)
            for combined, n_sim, a_sim, c_idx in cand_scores[:self.name_top_k]:
                cid = self.target_ids[c_idx]
                csrc = self.target_sources[c_idx]
                reasons = "exact_core" if q_n == self.target_names[c_idx] else "inverted_match"
                num_methods = 2 if (q_n == self.target_names[c_idx] or combined > 0.6) else 1

                results.append((
                    s1_id, cid, csrc, float(n_sim), float(a_sim), reasons, num_methods
                ))

        return results

    def _query_tfidf(
        self,
        query_df: pd.DataFrame,
        batch_size: int = 1000
    ) -> List[Tuple[str, str, str, float, float, str, int]]:
        """Original TF-IDF matrix dot-product candidate retrieval."""
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

            q_mat_name = self.name_vectorizer.transform(b_cores)
            sim_name = q_mat_name.dot(self.target_matrix_name.T)

            q_mat_addr = self.addr_vectorizer.transform(b_addrs)
            sim_addr = q_mat_addr.dot(self.target_matrix_addr.T)

            for i, s1_id in enumerate(b_ids):
                candidate_dict = {}
                q_core = b_cores[i].strip()
                q_addr = b_addrs[i].strip()

                if q_core and q_core in self.target_exact_map:
                    exact_indices = self.target_exact_map[q_core]
                    if len(exact_indices) > self.name_top_k:
                        ranked_indices = sorted(
                            exact_indices,
                            key=lambda idx: fuzz.ratio(q_addr, self.target_addrs[idx]) if q_addr else 0,
                            reverse=True
                        )[:self.name_top_k]
                    else:
                        ranked_indices = exact_indices

                    for t_idx in ranked_indices:
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

                for cid, c_info in candidate_dict.items():
                    reasons_str = "+".join(sorted(c_info["reasons"]))
                    num_methods = len(c_info["reasons"])
                    results.append((
                        s1_id,
                        cid,
                        c_info["candidate_source"],
                        c_info["name_score"],
                        c_info["addr_score"],
                        reasons_str,
                        num_methods
                    ))

        return results
