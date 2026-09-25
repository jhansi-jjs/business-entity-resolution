"""Feature engineering module for Business Entity Resolution Challenge.

Extracts pairwise similarity features between Source 1 records and candidate
records (Source 2 / Source 3), including multilingual/ASCII cross-script matching,
address numeric overlap, and candidate metadata.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.normalize import normalize_name, normalize_address, extract_postal_code

FEATURE_NAMES = [
    # Name features
    "feat_name_exact",
    "feat_name_core_exact",
    "feat_name_fuzz_ratio",
    "feat_name_ascii_fuzz_ratio",
    "feat_name_partial_ratio",
    "feat_name_token_sort_ratio",
    "feat_name_token_set_ratio",
    "feat_name_jaccard",
    "feat_name_containment",
    "feat_name_shared_tokens",
    "feat_name_first_token_match",
    "feat_name_last_token_match",
    "feat_name_len_diff",
    "feat_name_rel_len_diff",

    # Address features
    "feat_addr_exact",
    "feat_addr_fuzz_ratio",
    "feat_addr_partial_ratio",
    "feat_addr_token_sort_ratio",
    "feat_addr_token_set_ratio",
    "feat_addr_jaccard",
    "feat_addr_containment",
    "feat_addr_numeric_shared",
    "feat_addr_numeric_jaccard",
    "feat_addr_first_num_match",
    "feat_addr_postal_match",
    "feat_addr_len_diff",

    # Meta / Retrieval signals
    "feat_cand_is_s3",
    "feat_addr_is_empty",
    "feat_name_retrieval_score",
    "feat_addr_retrieval_score",
    "feat_num_blocking_methods",
]


def token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    u = tokens1.union(tokens2)
    return len(tokens1.intersection(tokens2)) / len(u) if u else 0.0


def token_containment(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Compute containment: fraction of smaller set contained in larger set."""
    if not tokens1 or not tokens2:
        return 0.0
    min_len = min(len(tokens1), len(tokens2))
    return len(tokens1.intersection(tokens2)) / min_len if min_len > 0 else 0.0


def compute_pair_features(
    s1_row: Dict[str, Any],
    cand_row: Dict[str, Any],
    name_retrieval_score: float = 0.0,
    addr_retrieval_score: float = 0.0,
    num_blocking_methods: int = 1,
) -> Dict[str, float]:
    """Compute all pairwise similarity features for a single (S1, Candidate) pair."""
    s1_norm = s1_row.get("name_norm", "")
    s1_core = s1_row.get("name_core", "")
    s1_ascii = s1_row.get("name_ascii", "")
    cand_norm = cand_row.get("name_norm", "")
    cand_core = cand_row.get("name_core", "")
    cand_ascii = cand_row.get("name_ascii", "")

    s1_addr = s1_row.get("addr_norm", "")
    cand_addr = cand_row.get("addr_norm", "")

    # Safeguard against treating two empty names as similar
    if not s1_norm or not cand_norm:
        exact_name = 0.0
        exact_core = 0.0
        fuzz_ratio = 0.0
        ascii_fuzz_ratio = 0.0
        partial_ratio = 0.0
        token_sort = 0.0
        token_set = 0.0
        jaccard_name = 0.0
        containment_name = 0.0
        shared_tokens = 0.0
        first_match = 0.0
        last_match = 0.0
        name_len_diff = float(abs(len(s1_norm) - len(cand_norm)))
        name_rel_diff = 1.0
    else:
        s1_tokens = set(s1_norm.split())
        cand_tokens = set(cand_norm.split())

        exact_name = 1.0 if s1_norm == cand_norm else 0.0
        exact_core = 1.0 if s1_core and s1_core == cand_core else 0.0

        fuzz_ratio = fuzz.ratio(s1_norm, cand_norm) / 100.0
        ascii_fuzz_ratio = fuzz.ratio(s1_ascii, cand_ascii) / 100.0 if (s1_ascii and cand_ascii) else fuzz_ratio
        partial_ratio = fuzz.partial_ratio(s1_norm, cand_norm) / 100.0
        token_sort = fuzz.token_sort_ratio(s1_norm, cand_norm) / 100.0
        token_set = fuzz.token_set_ratio(s1_norm, cand_norm) / 100.0

        jaccard_name = token_jaccard(s1_tokens, cand_tokens)
        containment_name = token_containment(s1_tokens, cand_tokens)
        shared_tokens = float(len(s1_tokens.intersection(cand_tokens)))

        s1_tok_list = s1_norm.split()
        cand_tok_list = cand_norm.split()
        first_match = 1.0 if (s1_tok_list and cand_tok_list and s1_tok_list[0] == cand_tok_list[0]) else 0.0
        last_match = 1.0 if (s1_tok_list and cand_tok_list and s1_tok_list[-1] == cand_tok_list[-1]) else 0.0

        len1, len2 = len(s1_norm), len(cand_norm)
        name_len_diff = float(abs(len1 - len2))
        name_rel_diff = name_len_diff / max(1, max(len1, len2))

    # Address features
    addr_empty = 1.0 if not cand_addr or cand_addr.strip() == "" else 0.0

    if not addr_empty and s1_addr:
        exact_addr = 1.0 if s1_addr == cand_addr else 0.0
        addr_fuzz = fuzz.ratio(s1_addr, cand_addr) / 100.0
        addr_partial = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        addr_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

        s1_a_tokens = set(s1_addr.split())
        cand_a_tokens = set(cand_addr.split())
        addr_jaccard = token_jaccard(s1_a_tokens, cand_a_tokens)
        addr_contain = token_containment(s1_a_tokens, cand_a_tokens)

        s1_nums = set(s1_row.get("addr_nums", []))
        cand_nums = set(cand_row.get("addr_nums", []))

        shared_nums = float(len(s1_nums.intersection(cand_nums)))
        num_jaccard = token_jaccard(s1_nums, cand_nums)

        s1_num_list = s1_row.get("addr_nums", [])
        cand_num_list = cand_row.get("addr_nums", [])
        first_num = 1.0 if (s1_num_list and cand_num_list and s1_num_list[0] == cand_num_list[0]) else 0.0

        p1 = s1_row.get("postal_code")
        p2 = cand_row.get("postal_code")
        postal_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0

        addr_len_diff = float(abs(len(s1_addr) - len(cand_addr)))
    else:
        exact_addr = 0.0
        addr_fuzz = 0.0
        addr_partial = 0.0
        addr_sort = 0.0
        addr_set = 0.0
        addr_jaccard = 0.0
        addr_contain = 0.0
        shared_nums = 0.0
        num_jaccard = 0.0
        first_num = 0.0
        postal_match = 0.0
        addr_len_diff = 0.0

    cid = str(cand_row.get("entity_id", ""))
    is_s3 = 1.0 if cid.startswith("S3-") else 0.0

    return {
        "feat_name_exact": exact_name,
        "feat_name_core_exact": exact_core,
        "feat_name_fuzz_ratio": fuzz_ratio,
        "feat_name_ascii_fuzz_ratio": ascii_fuzz_ratio,
        "feat_name_partial_ratio": partial_ratio,
        "feat_name_token_sort_ratio": token_sort,
        "feat_name_token_set_ratio": token_set,
        "feat_name_jaccard": jaccard_name,
        "feat_name_containment": containment_name,
        "feat_name_shared_tokens": shared_tokens,
        "feat_name_first_token_match": first_match,
        "feat_name_last_token_match": last_match,
        "feat_name_len_diff": name_len_diff,
        "feat_name_rel_len_diff": name_rel_diff,

        "feat_addr_exact": exact_addr,
        "feat_addr_fuzz_ratio": addr_fuzz,
        "feat_addr_partial_ratio": addr_partial,
        "feat_addr_token_sort_ratio": addr_sort,
        "feat_addr_token_set_ratio": addr_set,
        "feat_addr_jaccard": addr_jaccard,
        "feat_addr_containment": addr_contain,
        "feat_addr_numeric_shared": shared_nums,
        "feat_addr_numeric_jaccard": num_jaccard,
        "feat_addr_first_num_match": first_num,
        "feat_addr_postal_match": postal_match,
        "feat_addr_len_diff": addr_len_diff,

        "feat_cand_is_s3": is_s3,
        "feat_addr_is_empty": addr_empty,
        "feat_name_retrieval_score": float(name_retrieval_score),
        "feat_addr_retrieval_score": float(addr_retrieval_score),
        "feat_num_blocking_methods": float(num_blocking_methods),
    }


def enrich_record_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to precalculate normalized forms and tokens for high-speed feature creation."""
    name = row.get("business_name", "")
    addr = row.get("business_address", "")
    cntry = row.get("country", "")

    norm_n, core_n, ascii_n = normalize_name(name)
    norm_a, nums = normalize_address(addr)
    postal = extract_postal_code(nums, cntry)

    return {
        "entity_id": str(row.get("entity_id", "")),
        "country": str(cntry),
        "name_norm": norm_n,
        "name_core": core_n,
        "name_ascii": ascii_n,
        "addr_norm": norm_a,
        "addr_nums": nums,
        "postal_code": postal,
    }
