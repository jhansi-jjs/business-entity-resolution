"""Data loader module for Business Entity Resolution Challenge.

Loads, validates, and standardizes ingestion of Source 1, 2, 3 TSV files
and ground truth annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Iterator
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def get_default_paths(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Return default paths for all organizer datasets."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    return {
        "train_source1": base_dir / "dataset" / "train" / "train_source1.tsv",
        "train_source2": base_dir / "dataset" / "train" / "train_source2.tsv",
        "train_source3": base_dir / "dataset" / "train" / "train_source3.tsv",
        "train_gt": base_dir / "dataset" / "train" / "train_ground_truth.tsv",
        "test_source1": base_dir / "dataset" / "test" / "test_source1.tsv",
        "test_source2": base_dir / "dataset" / "test" / "test_source2.tsv",
        "test_source3": base_dir / "dataset" / "test" / "test_source3.tsv",
    }


def validate_source_dataframe(
    df: pd.DataFrame,
    source_name: str,
    expected_prefix: str,
    strict_uniqueness: bool = True
) -> None:
    """Validate schema, prefixes, and uniqueness of entity IDs in a source DataFrame."""
    missing_cols = set(EXPECTED_SOURCE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{source_name} is missing required columns: {missing_cols}")

    if strict_uniqueness:
        num_dups = df["entity_id"].duplicated().sum()
        if num_dups > 0:
            raise ValueError(f"{source_name} has {num_dups} duplicate entity_ids!")

    non_matching_prefix = ~df["entity_id"].astype(str).str.startswith(expected_prefix)
    if non_matching_prefix.any():
        sample_bad = df[non_matching_prefix]["entity_id"].head(5).tolist()
        raise ValueError(
            f"{source_name} contains entity IDs that do not start with {expected_prefix}: {sample_bad}"
        )


def load_source_tsv(
    path: Union[str, Path],
    source_name: str,
    expected_prefix: str,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Load a single source TSV file with strict validation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")

    logger.info(f"Loading {source_name} from {path} (nrows={nrows})...")
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        usecols=usecols,
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string"
        },
        keep_default_na=False
    )
    df.columns = [c.strip() for c in df.columns]

    if validate and usecols is None:
        validate_source_dataframe(df, source_name=source_name, expected_prefix=expected_prefix)

    logger.info(f"Loaded {source_name}: {len(df):,} rows.")
    return df


def iter_source_tsv(
    path: Union[str, Path],
    chunksize: int = 100000,
    usecols: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    """Iterate over a source TSV in chunks to prevent high memory usage."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=chunksize,
        usecols=usecols,
        dtype={
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string"
        },
        keep_default_na=False
    ):
        chunk.columns = [c.strip() for c in chunk.columns]
        yield chunk


def load_ground_truth(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Load training ground truth TSV."""
    if path is None:
        path = get_default_paths()["train_gt"]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    logger.info(f"Loading ground truth from {path} (nrows={nrows})...")
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype={"source1_entity_id": "string", "matched_entity_ids": "string"},
        keep_default_na=False
    )
    df.columns = [c.strip() for c in df.columns]

    missing_cols = set(EXPECTED_GT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Ground truth is missing required columns: {missing_cols}")

    if validate:
        num_dups = df["source1_entity_id"].duplicated().sum()
        if num_dups > 0:
            raise ValueError(f"Ground truth has {num_dups} duplicate source1_entity_ids!")

        non_s1 = ~df["source1_entity_id"].astype(str).str.startswith("S1-")
        if non_s1.any():
            raise ValueError(f"Ground truth contains non-S1 IDs: {df[non_s1].head(5).tolist()}")

    logger.info(f"Loaded ground truth: {len(df):,} rows.")
    return df


def load_training_data(
    base_dir: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training datasets and ground truth."""
    paths = get_default_paths(base_dir)
    s1 = load_source_tsv(paths["train_source1"], "Source 1 (Train)", "S1-", nrows=nrows, validate=validate)
    s2 = load_source_tsv(paths["train_source2"], "Source 2 (Train)", "S2-", nrows=nrows, validate=validate)
    s3 = load_source_tsv(paths["train_source3"], "Source 3 (Train)", "S3-", nrows=nrows, validate=validate)
    gt = load_ground_truth(paths["train_gt"], nrows=nrows, validate=validate)
    return s1, s2, s3, gt


def load_test_data(
    base_dir: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
    validate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test datasets."""
    paths = get_default_paths(base_dir)
    s1 = load_source_tsv(paths["test_source1"], "Source 1 (Test)", "S1-", nrows=nrows, validate=validate)
    s2 = load_source_tsv(paths["test_source2"], "Source 2 (Test)", "S2-", nrows=nrows, validate=validate)
    s3 = load_source_tsv(paths["test_source3"], "Source 3 (Test)", "S3-", nrows=nrows, validate=validate)
    return s1, s2, s3


if __name__ == "__main__":
    paths = get_default_paths()
    print("Checking default paths:")
    for k, p in paths.items():
        print(f"  {k}: exists={p.is_file()} ({p.name})")
