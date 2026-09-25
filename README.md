# business-entity-resolution
Production-grade ML pipeline for business entity resolution and record linkage.

---

## Business Entity Resolution Solution & Pipeline

This repository implements a high-precision, scalable entity resolution and record linkage system designed to link reference businesses from **Source 1** to matching records across noisy **Source 2** and **Source 3** datasets without external APIs, paid cloud services, or web lookups.

### Architecture Overview

```text
RAW ORGANIZER TSVs (S1, S2, S3)
        │
        ▼
[1] DATA LOADER & SCHEMA VALIDATION (src/data_loader.py)
        │
        ▼
[2] DETERMINISTIC NORMALIZATION (src/normalize.py)
        ├── Business Name Normalization & Core Extraction
        ├── Address Standardisation & Numeric Token Extraction
        └── Legal Suffix & Multi-Script (Indic, French, US) Handling
        │
        ▼
[3] SCALABLE CANDIDATE GENERATION / BLOCKING (src/blocking.py)
        ├── Lossless Country Partitioning (US, India, France)
        ├── Exact Core Inverted Index
        ├── Prefix-4 & Token 2-Gram Inverted Index
        ├── Rare Token & Postal Code Index
        └── Inverted-Index Multi-Strategy Retrieval (400–800+ q/s on CPU)
        │
        ▼
[4] PAIRWISE FEATURE ENGINEERING (src/features.py)
        ├── 14 Name Similarity Features (Exact, Fuzz, Token Sort/Set, Indic/ASCII)
        ├── 12 Address Similarity Features (Numeric Jaccard, Postal, Fuzz)
        └── 5 Metadata & Retrieval Scores (Throughput: ~35,000 pairs/sec)
        │
        ▼
[5] MATCHING MODEL (src/train.py)
        └── HistGradientBoosting Classifier (Grouped Validation by S1)
        │
        ▼
[6] THRESHOLD TUNING (F0.5 Optimized)
        └── Empirically calibrated threshold = 0.75
        │
        ▼
[7] MEMORY-BOUNDED INFERENCE & AGGREGATION (src/predict.py)
        ├── Lightweight tuple lookup & on-demand candidate enrichment (<4.5 GB RAM)
        ├── Atomic file writing (.tmp promotion)
        ├── output/matching_results.tsv (scored leaderboard file)
        └── output/candidate_pairs.tsv (blocking candidate set)
        │
        ▼
[8] SUBMISSION VALIDATION (utils/validate_submission.py)
        └── Official Organizer Validator (PASS)
```

---

### Repository Structure

```text
business-entity-resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py         # Organizer submission validator
├── Documentation_template.md          # Completed methodology documentation
├── README.md                          # Project documentation
├── requirements.txt                   # Minimal pinned dependencies
├── run.py                             # Main reproduction CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # TSV ingestion & schema validation
│   ├── normalize.py                   # Multi-script name & address cleaner
│   ├── blocking.py                    # Multi-key Inverted Index Blocker
│   ├── features.py                    # 31 pairwise similarity features
│   ├── train.py                       # Grouped validation & GBDT training
│   ├── evaluate.py                    # Entity-level macro F0.5 metrics
│   ├── predict.py                     # Memory-bounded inference & TSV writer
│   └── pipeline.py                    # Full 8-stage pipeline orchestrator
├── models/
│   ├── matcher.joblib                 # Trained GBDT matcher
│   └── model_config.json              # Model hyperparameters & threshold (0.75)
├── output/
│   ├── matching_results.tsv           # Final matches for leaderboard scoring
│   ├── candidate_pairs.tsv            # Candidate blocking pairs
│   └── sample_smoke/                  # Isolated smoke test outputs
├── tests/
│   ├── test_evaluation_regression.py  # Unit tests for Macro F0.5 & singletons
│   └── test_normalization_regression.py # Unit tests for legal suffixes & Indic text
└── experiments/
    ├── eda_report.json                # Complete structural audit
    ├── experiments.csv                # Model comparison results
    ├── ablation_results.csv           # Feature ablation studies
    ├── assessment_evaluation.json     # Frozen held-out assessment metrics
    ├── full_index_tune_benchmark.json # 10.3M uncurated target benchmark
    └── splits/
        └── split_manifest.json        # Stratified leak-free train/tune/test splits
```

---

### Empirical Validation & Benchmark Results

All metrics below are measured on leak-free splits sampled strictly from organizer training data.

#### 1. Validated Test Submissions Ready for Portal Upload

| Submission Version | File Path | Row Count | File Size | SHA-256 Checksum | Validator Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GBDT Matching Model (v1)** | `output/matching_results.tsv`<br>`output/submission_v1_gbdt/matching_results.tsv` | **1,732,544** | 162.4 MB | `947F5F5C5E05F27967D5DEE310E3D884C02C4B54DA39089E0AF3A629FD9D609E` | **PASS** (`--check-ids` verified) | 10.73M links, 19,371 genuine singletons (1.12%). Matcher threshold = 0.75 |
| **Deterministic Baseline** | `output/baseline_matching_results.tsv` | **1,732,544** | 261.8 MB | `C7D40626FE75425EB6763BEA6598EF1F1BC64722ED8B9EC53610B79CE8816841` | **PASS** | Initial rule-based baseline (scored 0.386 on public leaderboard) |

#### 2. Empirical Benchmark: Full-Index Retrieval & Realistic Hard Negatives
To diagnose why models evaluated against curated target pools (25k targets) fail on multi-million uncurated indexes, we conducted a strictly leak-free benchmark on the complete **4,133,346 India training target index** ([`experiments/train_with_hard_negatives.py`](experiments/train_with_hard_negatives.py)):

| Metric | Model Trained on Curated Targets | Model Trained on Realistic Full-Target Negatives | Measured Delta |
| :--- | :---: | :---: | :---: |
| **Macro $F_{0.5}$ (threshold 0.65)** | **0.4995** | **0.7744** | **+0.2749 (+55.0%)** |
| **Precision** | **49.73%** | **87.77%** | **+38.04%** |
| **Recall (Non-Singletons)** | 60.44% | 59.82% | -0.62% |
| **Singleton Accuracy (Rejection)** | **0.0%** (100% False Positives) | **100.0%** (0% False Positives) | **+100.0%** |
| **Avg. Matches Predicted / Query** | **5.58** (Overprediction) | **2.34** (Controlled) | Ground-truth avg is ~3.67 |

**Core Diagnostic Finding:** Training against small curated target pools prevents models from learning discriminative negative boundaries against multi-million lookalikes, leading to 100% false-positive rates on singletons and severe precision collapse. Training on realistic negatives retrieved directly from the full target pool restores precision to **87.77%** and yields **100% singleton rejection**.

#### 3. Uncurated 10.3M Full Target Index Blocking Benchmark
Evaluated against the complete uncurated training target pool (10.3M records in Source 2 + Source 3):
- **Overall Candidate Recall:** `95.65%`
- **India Partition Recall (4.13M targets):** `94.79%`
- **US Partition Recall (6.19M targets):** `96.59%`
- **Average Candidates per Query:** `32.4`
- **Retrieval Latency:** `1.2–2.4 ms/query` (`419–816 queries/sec` on CPU)

---

### Hardware Efficiency & Scalability

- **Zero GPU Requirement:** Complete candidate generation, feature extraction, and GBDT inference execute entirely on CPU.
- **Strict Memory Bounds:** Uses on-demand record dictionary enrichment and tuple-based target indexes, keeping peak memory working set below `4.5 GB RAM` (well within 16 GB laptop limits).
- **Fast Inverted Index:** Inverted index with exact core, prefix-4, and token 2-grams replaces $O(N \cdot M)$ brute force sparse dot products, achieving up to 800+ queries/second per CPU thread.

---

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/jhansi-jjs/business-entity-resolution.git
cd business-entity-resolution

# Install required dependencies
pip install -r requirements.txt
```

---

### Generating Submission Files

#### 1. Fast Smoke Test (200 Queries)
Runs an isolated smoke test to verify candidate generation, scoring, and output writing:
```bash
python run.py --sample 200
```
*Outputs are saved to `output/sample_smoke/` without touching production submission files.*

#### 2. Full Test Inference (All 1,732,544 Entities)
Processes all 1.73M Source 1 test entities across US, India, and France:
```bash
python run.py
```
*Outputs are saved to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.*

---

### Running the Organizer Validator Independently

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The script verifies:
1. File existence and valid TSV header format (`entity_id`, `matched_entity_ids`).
2. Exact 1-to-1 match with test Source 1 entity IDs (no duplicates, no omissions, no extra IDs).
3. Candidate subset invariant: every predicted match is strictly present in the candidate set.
4. Correct comma-separated string representation for multiple matches and empty strings for singletons.

---

### Running Unit Tests

```bash
pytest tests/ -v
```
All unit tests for Macro $F_{0.5}$ metric computation, singleton edge cases, address normalization, and Indic/French entity handling pass without warnings.

