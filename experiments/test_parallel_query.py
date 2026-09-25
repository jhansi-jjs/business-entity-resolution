"""Test parallel query throughput with joblib on 500 queries."""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking import CountryBlocker
from src.normalize import normalize_name, normalize_address
from src.data_loader import get_default_paths

def test_parallel():
    paths = get_default_paths()
    s2 = pd.read_csv(paths["train_source2"], sep="\t", nrows=100000)
    s2["name_core"] = [normalize_name(n)[1] for n in s2["business_name"]]
    s2["addr_norm"] = [normalize_address(a)[0] for a in s2["business_address"]]

    blocker = CountryBlocker(country="US", backend="inverted")
    blocker.fit(s2)

    queries = pd.read_csv(paths["train_source1"], sep="\t", nrows=500)
    queries["name_core"] = [normalize_name(n)[1] for n in queries["business_name"]]
    queries["addr_norm"] = [normalize_address(a)[0] for a in queries["business_address"]]

    # Single-thread
    t0 = time.time()
    cands_single = blocker.query(queries)
    t_single = time.time() - t0
    print(f"Single-thread: {len(queries)} queries in {t_single:.3f}s ({len(queries)/t_single:.1f} q/s)")

    # 4-thread / joblib chunking
    chunk_size = 125
    chunks = [queries.iloc[i:i+chunk_size] for i in range(0, len(queries), chunk_size)]

    t0 = time.time()
    cands_parallel = Parallel(n_jobs=4, prefer="threads")(
        delayed(blocker.query)(chunk) for chunk in chunks
    )
    all_cands = [item for sub in cands_parallel for item in sub]
    t_par = time.time() - t0
    print(f"Parallel (4 threads): {len(queries)} queries in {t_par:.3f}s ({len(queries)/t_par:.1f} q/s)")
    print(f"Speedup: {t_single/t_par:.2f}x")
    print(f"Results match count: {len(cands_single)} vs {len(all_cands)}")

if __name__ == "__main__":
    test_parallel()
