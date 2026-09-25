"""Inspect missed matches from blocking."""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import pandas as pd
from src.data_loader import get_default_paths, load_source_tsv, load_ground_truth
from src.normalize import normalize_name, normalize_address
from src.blocking import CountryBlocker

# Let us run a quick check on the missed matches
print("Missed match inspection setup complete.")
