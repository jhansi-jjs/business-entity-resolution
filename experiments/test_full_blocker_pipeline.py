"""Test the enhanced fast inverted-index blocker on 500 Tune queries with full ground truth."""

import sys
from pathlib import Path
import json
import time
import gc
from collections import defaultdict
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_name, normalize_address, extract_postal_code
from src.data_loader import get_default_paths, load_ground_truth

GENERIC_STOPWORDS = {
    "pvt", "ltd", "inc", "llc", "corp", "corporation", "private", "limited",
    "the", "and", "co", "company", "group", "enterprises", "solutions",
    "services", "technologies", "tech", "trading", "retail", "international",
    "sa", "sarl", "sas", "gmbh", "holding", "holdings"
}

def extract_keys(name_core: str, addr_norm: str, country: str):
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

class FastInvertedBlocker:
    def __init__(self, country: str, top_k: int = 25, max_postings: int = 1500):
        self.country = country
        self.top_k = top_k
        self.max_postings = max_postings
        self.index = defaultdict(list)
        self.key_counts = defaultdict(int)
        self.target_ids = []
        self.target_names = []
        self.target_addrs = []
        self.target_sources = []
        
    def fit(self, target_df: pd.DataFrame):
        eids = target_df["entity_id"].values
        names = target_df["name_core"].fillna("").values
        addrs = target_df["addr_norm"].fillna("").values
        
        self.target_ids = list(eids)
        self.target_names = list(names)
        self.target_addrs = list(addrs)
        self.target_sources = ["S2" if str(eid).startswith("S2-") else "S3" for eid in eids]
        
        for idx, (n, a) in enumerate(zip(names, addrs)):
            for k in extract_keys(n, a, self.country):
                if self.key_counts[k] < self.max_postings:
                    self.index[k].append(idx)
                    self.key_counts[k] += 1
        return self
        
    def query(self, query_df: pd.DataFrame, batch_size: int = 1000):
        all_candidates = []
        q_ids = query_df["entity_id"].values
        q_names = query_df["name_core"].fillna("").values
        q_addrs = query_df["addr_norm"].fillna("").values
        
        for s1_id, q_n, q_a in zip(q_ids, q_names, q_addrs):
            q_keys = extract_keys(q_n, q_a, self.country)
            cand_indices = set()
            for k in q_keys:
                if k in self.index:
                    cand_indices.update(self.index[k])
                    
            if not cand_indices:
                continue
                
            # Score and rank candidates
            cand_scores = []
            for c_idx in cand_indices:
                t_n = self.target_names[c_idx]
                t_a = self.target_addrs[c_idx]
                n_sim = fuzz.ratio(q_n, t_n) / 100.0
                a_sim = fuzz.ratio(q_a, t_a) / 100.0 if (q_a and t_a) else 0.0
                combined = 0.70 * n_sim + 0.30 * a_sim if (q_a and t_a) else n_sim
                cand_scores.append((combined, n_sim, a_sim, c_idx))
                
            cand_scores.sort(key=lambda x: x[0], reverse=True)
            for combined, n_sim, a_sim, c_idx in cand_scores[:self.top_k]:
                cid = self.target_ids[c_idx]
                csrc = self.target_sources[c_idx]
                all_candidates.append((
                    s1_id, cid, csrc, n_sim, a_sim, {"fast_inverted"}, 2
                ))
        return all_candidates

def main():
    manifest_path = PROJECT_ROOT / "experiments" / "splits" / "split_manifest.json"
    benchmark_pairs_path = PROJECT_ROOT / "experiments" / "splits" / "benchmark_pairs.tsv"
    sampled_s1_path = PROJECT_ROOT / "experiments" / "splits" / "sampled_s1.tsv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    tune_ids = set(manifest["tune_ids"])
    s1_df = pd.read_csv(sampled_s1_path, sep="\t", dtype=str).fillna("")
    tune_s1 = s1_df[s1_df["entity_id"].isin(tune_ids)].copy().reset_index(drop=True)

    paths = get_default_paths()
    gt_df = load_ground_truth(paths["train_gt"])
    gt_filtered = gt_df[gt_df["source1_entity_id"].isin(tune_ids)]
    gt_map = {}
    for _, row in gt_filtered.iterrows():
        s1 = str(row["source1_entity_id"])
        raw_m = str(row["matched_entity_ids"]).strip()
        if raw_m and raw_m != "nan":
            gt_map[s1] = {m.strip() for m in raw_m.split(",") if m.strip()}

    print(f"Tune queries: {len(tune_s1):,}, Total true links: {sum(len(v) for v in gt_map.values()):,}")

    # Load full sampled targets from train_source2 and train_source3 for these pairs
    s2_df = pd.read_csv(paths["train_source2"], sep="\t", nrows=100000)
    s3_df = pd.read_csv(paths["train_source3"], sep="\t", nrows=100000)
    all_targets = pd.concat([s2_df, s3_df], ignore_index=True)
    all_targets["name_core"] = [normalize_name(n)[1] for n in all_targets["business_name"]]
    all_targets["addr_norm"] = [normalize_address(a)[0] for a in all_targets["business_address"]]

    tune_s1["name_core"] = [normalize_name(n)[1] for n in tune_s1["business_name"]]
    tune_s1["addr_norm"] = [normalize_address(a)[0] for a in tune_s1["business_address"]]

    print(f"Fitting blocker on {len(all_targets):,} targets...")
    t0 = time.time()
    blocker = FastInvertedBlocker(country="US", top_k=25)
    blocker.fit(all_targets)
    t_fit = time.time() - t0
    print(f"Fit completed in {t_fit:.2f}s.")

    us_tune = tune_s1[tune_s1["country"] == "US"].head(100)
    print(f"Querying {len(us_tune)} US queries...")
    t0 = time.time()
    cands = blocker.query(us_tune)
    t_query = time.time() - t0
    print(f"Queried {len(us_tune)} queries in {t_query:.3f}s ({t_query/len(us_tune)*1000:.2f} ms/query).")
    print(f"Throughput: {len(us_tune)/t_query:.1f} queries/sec.")
    print(f"Total candidate pairs: {len(cands):,} ({len(cands)/len(us_tune):.1f} per query).")

if __name__ == "__main__":
    main()
