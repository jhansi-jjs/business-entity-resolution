# business-entity-resolution
ML pipeline for business entity resolution and record linkage

---

## Business Entity Resolution Solution & Pipeline

This repository implements a high-precision, scalable entity resolution and record linkage system designed to link reference businesses from **Source 1** to matching records across noisy **Source 2** and **Source 3** datasets.

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
        └── Address Standardisation & Numeric Token Extraction
        │
        ▼
[3] CANDIDATE GENERATION / BLOCKING (src/blocking.py)
        ├── Lossless Country Partitioning (US, India, France)
        ├── Character n-gram TF-IDF (3, 5) Top-K Retrieval
        ├── Exact Core Inverted Index
        └── Multi-Strategy Candidate Union (>98.9% Recall)
        │
        ▼
[4] PAIRWISE FEATURE ENGINEERING (src/features.py)
        ├── 13 Name Similarity Features (Exact, Fuzz, Token, Jaccard, Containment)
        ├── 12 Address Similarity Features (Numeric Jaccard, Postal, Fuzz)
        └── 5 Metadata & Retrieval Scores
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
[7] ENTITY-LEVEL SET AGGREGATION & SUBMISSION WRITING (src/predict.py)
        ├── output/matching_results.tsv (scored leaderboard file)
        └── output/candidate_pairs.tsv (blocking candidate set)
        │
        ▼
[8] SUBMISSION VALIDATION (utils/validate_submission.py)
        └── Official Organizer Validator (PASS)
```

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
│   ├── blocking.py                    # CountryBlocker & candidate union
│   ├── features.py                    # 30 pairwise similarity features
│   ├── train.py                       # Grouped validation & GBDT training
│   ├── evaluate.py                    # Entity-level macro F0.5 metrics
│   ├── predict.py                     # Inference & TSV generator
│   └── pipeline.py                    # Full 8-stage pipeline orchestrator
├── models/
│   ├── matcher.joblib                 # Trained GBDT matcher
│   └── model_config.json              # Model hyperparameters & threshold (0.75)
├── output/
│   ├── matching_results.tsv           # Final matches for leaderboard scoring
│   └── candidate_pairs.tsv            # Candidate blocking pairs
└── experiments/
    ├── eda_report.json                # Complete structural audit
    ├── experiments.csv                # Model comparison results
    └── ablation_results.csv           # Feature ablation studies
```

### Environment Setup

```bash
pip install -r requirements.txt
```

### Generating Submission Files

To run the complete pipeline and validate the submission outputs:
```bash
python run.py
```

### Running the Organizer Validator Independently

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Key Performance Metrics
- **Validation Macro $F_{0.5}$:** `0.9843`
- **Validation Macro Precision:** `0.9935`
- **Validation Macro Recall:** `0.9656`
- **Candidate Blocking Recall:** `98.97%`
- **Singleton Accuracy:** `100.0%`
- **Validator Result:** `PASS — no blocking issues found. Safe to submit.`
