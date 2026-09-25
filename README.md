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

#### 1. Frozen Held-Out Assessment Split (500 Queries, Evaluated Once)
- **Macro $F_{0.5}$:** `0.9893`
- **Macro Precision:** `0.9959`
- **Macro Recall:** `0.9748`
- **Singleton Accuracy:** `100.0%` (27 / 27 true singletons correctly assigned empty match sets)
- **Average Candidates per Query:** `29.4`

#### 2. Uncurated 10.3M Full Target Index Blocking Benchmark
Evaluated against the complete uncurated training target pool (10.3M records in Source 2 + Source 3):
- **Overall Candidate Recall:** `95.65%`
- **India Partition Recall (4.13M targets):** `94.79%`
- **US Partition Recall (6.19M targets):** `96.59%`
- **Average Candidates per Query:** `32.4`
- **Retrieval Latency:** `1.2–2.4 ms/query` (`419–816 queries/sec` on CPU)

#### 3. Model Comparison on Tuning Split (Macro $F_{0.5}$)
| Model Architecture | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Chosen)** | **0.9972** | **0.9723** | **0.9921** | **0.75** |
| Random Forest | 0.9958 | 0.9587 | 0.9881 | 0.70 |
| Logistic Regression | 0.9812 | 0.9529 | 0.9754 | 0.60 |
| Decision Tree | 0.9785 | 0.9560 | 0.9739 | 0.65 |

#### 4. Feature Ablation Analysis
Ablating feature subsets from the 31-feature set confirms complementary signals:
- **Baseline (All 31 features):** Macro $F_{0.5} = 0.9921$
- **Without Address Features:** Macro $F_{0.5} = 0.9234$ ($-0.0687$ drop)
- **Without Name Fuzzy Features:** Macro $F_{0.5} = 0.8841$ ($-0.1080$ drop)
- **Without Retrieval Scores:** Macro $F_{0.5} = 0.9798$ ($-0.0123$ drop)

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

