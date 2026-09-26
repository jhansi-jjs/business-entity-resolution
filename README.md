# Business Entity Resolution — Amazon ML Challenge 2026

Production-grade entity resolution and record linkage pipeline designed to link reference businesses from **Source 1** to matching records across noisy **Source 2** and **Source 3** datasets without external APIs, paid cloud services, or web lookups.

Optimized strictly for the official competition evaluation metric: **Macro-Averaged $F_{0.5}$** (penalizing false merges twice as heavily as missed links, with strict singleton credit).

---

## 1. System Architecture

```text
RAW TSVs (Source 1: 1.73M queries, Source 2: 4.89M targets, Source 3: 5.70M targets)
        │
        ▼
[1] DATA LOADER & SCHEMA VALIDATION (src/data_loader.py)
        │
        ▼
[2] DETERMINISTIC NORMALIZATION (src/normalize.py)
        ├── Business Name Normalization & Core Extraction
        ├── Address Standardization & Numeric Token Parsing
        ├── Postal Code Extraction (US 5-digit, India 6-digit PIN, France 5-digit)
        └── Indic Cross-Script Transliteration (Devanagari/Latin ASCII)
        │
        ▼
[3] SCALABLE CANDIDATE GENERATION / BLOCKING (src/blocking.py)
        ├── Lossless Country Partitioning (US, India, France)
        ├── Strategy 1: Exact Core Inverted Index (address-ranked if >15)
        ├── Strategy 2: Postal / Address-Anchored Locality Key (name-ranked)
        ├── Strategy 3: Multi-Key Fuzzy Retrieval (rarity-sorted, posting-pruned)
        └── Compact Candidate Pools: ~29.8 candidates / query (zero brute force)
        │
        ▼
[4] VECTORIZED PAIRWISE FEATURE ENGINEERING (src/features.py, src/predict.py)
        ├── 14 Name Similarity Features (Exact, Fuzz, Token Sort/Set, Indic/ASCII)
        ├── 12 Address Similarity Features (Numeric Jaccard, Postal, Fuzz)
        ├── 5 Metadata & Retrieval Signals
        └── Precomputed Query Tokens & Direct NumPy Float32 Matrix Generation
        │
        ▼
[5] MATCHING MODEL & THRESHOLDING (src/train.py, models/matcher.joblib)
        ├── HistGradientBoosting Classifier (trained on 685k hard negatives)
        └── Calibrated Decision Threshold = 0.68 (Macro F0.5 Optimized)
        │
        ▼
[6] MEMORY-BOUNDED INFERENCE & ATOMIC PROMOTION (src/predict.py)
        ├── Memory footprint < 7.2 GB RAM (runs completely on local CPU)
        ├── Strict Subset Invariant: matches ⊆ candidates asserted per entity
        ├── Atomic Promotion: writes .tmp first, verifies line counts, renames
        ├── output/matching_results.tsv (1,732,544 rows)
        └── output/candidate_pairs.tsv (1,732,544 rows)
        │
        ▼
[7] SUBMISSION VALIDATION & AUDIT (utils/validate_submission.py, utils/audit_submission.py)
        ├── Official Organizer Submission Validator (PASS)
        └── Distribution Fingerprint & Human Spot-Check Audit
```

---

## 2. Engineering Log & Root-Cause Diagnosis

During the competition, prior iterations scored significantly below expectations (**0.386** on the baseline and **0.453** on the initial GBDT). A thorough architectural and code audit diagnosed the exact root causes and led to systemic fixes:

### Issue A: The Toy 25k Sandbox Distractor Bias
- **Symptom:** A toy experiment on 25k queries claimed `0.9893` $F_{0.5}$, but the model failed on the real challenge.
- **Root Cause:** The 25k sandbox evaluated true links against only 10,000 random distractors where true answers were pre-selected. In the actual 10.3M target index, lookalike businesses with the same name in different cities or different branches in the same pin code overwhelmed the naive matcher.
- **Resolution:** Replaced sandbox data with **full-scale hard-negative mining** (`experiments/train_hard_negatives.py`), extracting 684,587 difficult candidate pairs directly through the blocker against 500,000 full targets per country.

### Issue B: The Baseline Single-Digit False Merge Bug (0.386 Score)
- **Symptom:** Deterministic baseline scored 0.386 with severe precision collapse.
- **Root Cause:** In `src/deterministic_baseline.py`, entities with matching core names were merged if *any single digit* matched between their addresses (e.g. `"Suite 1"` vs `"1st Ave"`). This created massive runaway mega-clusters of hundreds of unrelated entities.
- **Resolution:** Retired heuristic digit matching in favor of the 31-feature GBDT model requiring composite confidence across street tokens, house numbers, legal suffixes, and postal codes.

### Issue C: The 1,500 Postings Cap & Postal Code Bug (0.453 Score)
- **Symptom:** The initial GBDT scored 0.453 due to low recall.
- **Root Cause:** 
  1. `CountryBlocker` had a hard cap `max_postings = 1500` *during index creation*, literally discarding records from the index across 10 million entities.
  2. `extract_postal_code` in `src/normalize.py` expected a string, but received a list of numeric tokens. The regex check failed on every record, completely disabling postal code blocking keys.
- **Resolution:** Removed the index posting cap entirely, updated `extract_postal_code` to accept `Union[str, List[str]]`, and added Indic cross-script ASCII keys (`w1_asc`, `w2_asc`, `pref4_asc`) and address-anchored locality keys (`addr_num`).

### Issue D: Inference Throughput Bottleneck (8.4 q/s $\to$ ~300+ q/s)
- **Symptom:** Initial full test inference estimated 57+ hours execution time.
- **Root Cause:** 
  1. `enrich_record_dict` was repeatedly re-normalizing candidate records inside every batch (13.2 million redundant normalizations).
  2. In Strategy 3, generic English stopword keys (`"partners"`, `"center"`, `"care"`) had up to 86,000 postings, pulling 100,000 candidates for a single query into pure Python string loops.
- **Resolution:** 
  1. Vectorized upfront normalization of targets and precomputed query token sets.
  2. Rarity-sorted keys with selective posting pruning (keys with $>1,000$ postings skipped in Strategy 3).
  3. Feature computation directly into pre-allocated NumPy `float32` arrays.
  4. Throughput jumped from **8.4 q/s to ~300–400 q/s**, completing all 1.73M queries in **under 2 hours**.

---

## 3. Empirical Benchmark Results

### A. Held-Out Assessment Split (3,750 Queries against 500k Targets)
Evaluated on frozen held-out data with real ground-truth singleton rates and full-target distractors:

| Metric | Previous Baseline | Previous GBDT | **Final Retrained Model** |
| :--- | :---: | :---: | :---: |
| **Macro $F_{0.5}$** | 0.3860 | 0.4530 | **0.9274** |
| **Precision** | ~35.0% | ~48.2% | **96.20%** |
| **Recall** | ~60.0% | ~38.0% | **85.98%** |
| **Singleton Accuracy** | ~10.0% | ~52.0% | **92.31%** |
| **Decision Threshold** | Heuristic | 0.75 | **0.68** |

### B. Full 10.3M Target Index Candidate Retrieval (Blocking Benchmark)
- **US Recall (6.19M targets):** `93.75%` (165 / 176 true links)
- **India Recall (4.13M targets):** `76.04%` (146 / 192 true links)
- **Overall Recall:** `84.51%` (311 / 368 true links)
- **Average Candidates per Query:** `30.7`

---

## 4. Final Submission Statistics (Test Set)

Full inference across all **1,732,544 test entities** produced the official submission files:

- **Total S1 Queries Written:** `1,732,544`
- **Total Predicted Matches:** `6,059,456`
- **Average Links per Matched Query:** `3.84`
  - US: `3.58` links/query (matches ground truth registry density)
  - France: `6.35` links/query (matches SIREN/SIRET multi-branch density)
  - India: `2.51` links/query (matches standalone proprietor density)
- **Total Candidate Pairs:** `51,619,585` (**29.79 candidates / query**)
- **Singleton Rate:** `155,401 singletons (8.97%)`
- **Strict Subset Invariant Violations:** **0** (`matches ⊆ candidates` holds 100%)
- **Mega-Cluster Check:** **Zero** (95.3% of matched entities have 1 to 6 links; max cluster: 41)
- **Official Submission Validator Status:** **`PASS — no blocking issues found. Safe to submit.`**

---

## 5. Repository Structure

```text
business-entity-resolution/
├── dataset/                           # Ignored in git (organizer data)
│   ├── train/                         # train_source1, 2, 3, train_ground_truth
│   └── test/                          # test_source1, 2, 3
├── models/                            # Ignored in git (binary artifacts)
│   ├── matcher.joblib                 # Trained GBDT matcher (HistGradientBoosting)
│   └── model_config.json              # Model configuration & optimal threshold (0.68)
├── output/                            # Ignored in git (submission files)
│   ├── matching_results.tsv           # Official scored leaderboard submission
│   └── candidate_pairs.tsv            # Candidate blocking pairs
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # TSV ingestion & schema validation
│   ├── normalize.py                   # Multi-script name & address cleaner
│   ├── blocking.py                    # Multi-strategy Inverted Index Blocker
│   ├── features.py                    # 31 pairwise similarity features
│   ├── train.py                       # Base training routine
│   ├── evaluate.py                    # Macro F0.5 evaluation metrics
│   ├── predict.py                     # High-throughput vectorized inference engine
│   └── pipeline.py                    # End-to-end 8-stage pipeline orchestrator
├── utils/
│   ├── validate_submission.py         # Official organizer submission validator
│   └── audit_submission.py            # Quality audit & real test match spot-checker
├── experiments/
│   ├── train_hard_negatives.py        # Mining & training pipeline on 685k hard pairs
│   ├── benchmark_full_targets.py      # Full 10.3M target index blocking benchmark
│   └── splits/                        # Split manifest & sample queries
├── tests/
│   └── test_evaluation_regression.py  # Unit tests for Macro F0.5 & singletons
├── requirements.txt                   # Minimal pinned dependencies
├── run.py                             # Main reproduction CLI entrypoint
└── README.md
```

---

## 6. Setup & Execution

### Environment Installation
```bash
# Clone the repository
git clone https://github.com/jhansi-jjs/business-entity-resolution.git
cd business-entity-resolution

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing Full Inference
Processes all 1,732,544 test entities across US, France, and India, writing `output/matching_results.tsv` and `output/candidate_pairs.tsv` and running the official submission validator:
```bash
python run.py
```

### Running the Official Validator
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

### Running the Visual Spot-Check Audit
Pulls random test entities and prints side-by-side matches across Source 1, Source 2, and Source 3:
```bash
python utils/audit_submission.py --samples 15
```
