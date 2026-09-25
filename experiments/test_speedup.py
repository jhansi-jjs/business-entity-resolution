"""Test max_df and pruned TF-IDF speedup on target index."""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import normalize_name, normalize_address
from src.data_loader import get_default_paths, iter_source_tsv, load_ground_truth

def test_speed():
    paths = get_default_paths()

    # Load 100,000 sample targets
    print("Loading 100,000 target records...")
    targets = []
    for chunk in iter_source_tsv(paths["train_source2"], chunksize=100000):
        targets.append(chunk[chunk["country"] == "US"].head(50000))
        break
    for chunk in iter_source_tsv(paths["train_source3"], chunksize=100000):
        targets.append(chunk[chunk["country"] == "US"].head(50000))
        break
    target_df = pd.concat(targets, ignore_index=True)
    target_cores = [normalize_name(n)[1] for n in target_df["business_name"]]

    # Load 50 queries
    s1_df = pd.read_csv(paths["train_source1"], sep="\t", nrows=50)
    query_cores = [normalize_name(n)[1] for n in s1_df["business_name"]]

    # Baseline: no max_df
    t0 = time.time()
    vec_base = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, dtype=np.float32)
    mat_base = vec_base.fit_transform(target_cores)
    q_mat_base = vec_base.transform(query_cores)
    sim_base = q_mat_base.dot(mat_base.T)
    t_base = time.time() - t0
    print(f"Base: vocab={len(vec_base.vocabulary_):,}, nnz per row={sim_base.nnz / len(query_cores):.1f}, time={t_base:.2f}s")

    # Pruned: max_df=0.05, min_df=3
    t0 = time.time()
    vec_pruned = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_df=0.05, dtype=np.float32)
    mat_pruned = vec_pruned.fit_transform(target_cores)
    q_mat_pruned = vec_pruned.transform(query_cores)
    sim_pruned = q_mat_pruned.dot(mat_pruned.T)
    t_pruned = time.time() - t0
    print(f"Pruned: vocab={len(vec_pruned.vocabulary_):,}, nnz per row={sim_pruned.nnz / len(query_cores):.1f}, time={t_pruned:.2f}s")
    print(f"Speedup ratio: {t_base / max(0.01, t_pruned):.2f}x")

if __name__ == "__main__":
    test_speed()
