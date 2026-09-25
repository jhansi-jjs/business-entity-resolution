# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityLink Pro  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and verified an end-to-end Business Entity Resolution system that links reference businesses from Source 1 to zero, one, or multiple noisy records across Source 2 and Source 3. Our system implements lossless country-partitioned blocking, dual-signal character n-gram TF-IDF and exact core retrieval, a 31-dimensional pairwise feature space (including cross-script transliteration signals), and a HistGradientBoosting classifier optimized specifically for macro-averaged $F_{0.5}$.

On our strictly stratified, held-out assessment split (500 queries, zero data leakage), the frozen model achieves **0.9893 Macro $F_{0.5}$** (Macro Precision: **0.9959**, Macro Recall: **0.9748**, Singleton Accuracy: **100.0%** across 27 singletons). When evaluated against the entire uncurated training target index of **10,320,219 records** (without using ground truth labels to construct favorable pools), our blocking pipeline achieves **94.79% recall on India** (4.13M targets) and **96.59% recall on US** (6.19M targets), generating an average of 32.5 to 33.3 candidates per query.

---

## 2. Methodology

### 2.1 Problem Analysis
Audit and exploratory analysis across the organizer datasets revealed crucial structural characteristics:
- **Zero Cross-Country Linkage:** Empirical audit of all 7.6M true links in the ground truth confirmed exactly 0.0000% cross-country links. Entities strictly link within their jurisdiction (US, India, and France in test).
- **Scale & Asymmetry:** Training contains 2,206,821 Source 1 queries and 10,320,219 target records (5,034,616 in Source 2 and 5,285,603 in Source 3). The test set contains 1,732,544 Source 1 queries and ~10M targets. Full Cartesian comparison is mathematically and computationally intractable.
- **Multilingual Noise & Transliteration:** Real records contain mixed Indic scripts (Devanagari, Tamil, Telugu), Latin transliterations, variable legal suffixes (`Pvt Ltd`, `LLP`, `SARL`, `SCI`, `Corp`), trade name prefixes (`dba`, `t/a`), and missing addresses in ~3% of records.
- **Cardinality & Singletons:** 5.58% of reference entities are singletons (0 matches in S2/S3), while matched entities average ~3.7 links with a maximum of 11 links.

### 2.2 Solution Strategy
**Approach Type:** Multi-Strategy Country-Partitioned Blocking + Pairwise GBDT Matcher + Set Decision Thresholding.  
**Core Principles:** 
1. **Lossless Country Partitioning:** Slices the search space strictly by jurisdiction, eliminating impossible cross-border comparisons without recall loss.
2. **Dual-Signal Multi-Strategy Retrieval:** Unions character 3-5 gram TF-IDF on normalized name cores, character 3-5 gram TF-IDF on normalized addresses, and exact name core inverted index with address-aware ranking when exact duplicates exceed top-$K$.
3. **Macro $F_{0.5}$ Objective Optimization:** Because $F_{0.5}$ weights precision twice as heavily as recall, the decision threshold was tuned on a dedicated Tune split to 0.75, strictly suppressing false positive links while correctly preserving empty match sets for singletons.

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys & Retrieval Config:**
  - Country Partition: (India, US, France)
  - Strategy A: Character n-gram TF-IDF on normalized name core ($n \in [3, 5]$, sublinear TF, top-$K=25$, threshold=0.10)
  - Strategy B: Character n-gram TF-IDF on normalized address ($n \in [3, 5]$, sublinear TF, top-$K=8$, threshold=0.18)
  - Strategy C: Inverted index on exact normalized name core (ranked by address similarity when >25 exact candidates)
- **Candidate Pairs Generated:**
  - Full Target Index (India: 4,133,346 targets): Mean 32.52 pairs/query, Median 31.0, Range [26, 55], P95 44.55.
  - Full Target Index (US: 6,186,873 targets): Mean 33.26 pairs/query, Median 31.0, Range [27, 54], P95 52.00.
- **Measured Blocking Recall (Uncurated Full S2+S3 Target Index):**
  - **India:** **94.79%** (182 / 192 true links retrieved from 4.13M targets)
  - **US:** **96.59%** (170 / 176 true links retrieved from 6.19M targets)
  - **Overall True Link Retrieval:** **95.65%** without any favorable label filtering.

---

## 4. Matching Model

**Features Used (31 engineered features):**
- **Name Features (15):** Exact normalized match, exact name core match, RapidFuzz string ratio, partial ratio, token sort ratio, token set ratio, token Jaccard similarity, token containment, shared token count, first/last token match, character length difference, relative length difference, and ASCII transliteration fuzz ratio (`feat_name_ascii_fuzz_ratio`).
- **Address Features (14):** Exact normalized address match, RapidFuzz ratio, partial ratio, token sort ratio, token set ratio, token Jaccard, token containment, shared numeric tokens count, numeric token Jaccard, first numeric token match, postal/PIN code match, address missing indicator, and length difference.
- **Meta / Retrieval Provenance (2):** Name retrieval score, address retrieval score, and number of blocking strategies retrieving the pair.

**Model Type:** HistGradientBoosting Classifier (GBDT) with max depth 6 and 150 estimators.  
**Threshold Selection:** Grid search on the Tune split maximizing macro-averaged entity-level $F_{0.5}$. The selected threshold was **0.75**.

---

## 5. Results & Error Analysis

### 5.1 Model Comparison on Tune Split (14,742 Candidate Pairs across 500 S1 Queries)
| Model | Optimal Threshold | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Singleton Accuracy | Avg Matches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Heuristic Baseline | 0.55 | 0.9518 | 0.8536 | 0.9188 | 0.8929 | 3.10 |
| Logistic Regression | 0.50 | 0.9927 | 0.9791 | 0.9879 | 1.0000 | 3.48 |
| Random Forest | 0.50 | 0.9949 | 0.9774 | 0.9891 | 1.0000 | 3.46 |
| **HistGradientBoosting** | **0.75** | **0.9978** | **0.9785** | **0.9921** | **1.0000** | **3.45** |
| LightGBM | 0.85 | 0.9982 | 0.9761 | 0.9914 | 1.0000 | 3.44 |

### 5.2 Held-Out Assessment Evaluation (500 S1 Queries Evaluated Once)
- **Macro Precision:** **0.9959**
- **Macro Recall:** **0.9748**
- **Macro $F_{0.5}$:** **0.9893**
- **Singleton Accuracy:** **100.0%** (27 of 27 singletons correctly identified with 0 false matches)
- **Average Predicted Matches:** 3.43

**Breakdown by Country (Assessment Set):**
- **US (N=301):** Macro $F_{0.5}$ = **0.9934**, Precision = 0.9972, Recall = 0.9847, Singleton Acc = 1.0000
- **India (N=199):** Macro $F_{0.5}$ = **0.9831**, Precision = 0.9940, Recall = 0.9599, Singleton Acc = 1.0000

**Breakdown by Match Cardinality (Assessment Set):**
- **0 matches (Singleton, N=27):** Precision = 1.0000, Recall = 1.0000, $F_{0.5}$ = **1.0000**
- **1 match (N=27):** Precision = 1.0000, Recall = 1.0000, $F_{0.5}$ = **1.0000**
- **2-3 matches (N=206):** Precision = 0.9919, Recall = 0.9668, $F_{0.5}$ = **0.9837**
- **4+ matches (N=240):** Precision = 0.9985, Recall = 0.9760, $F_{0.5}$ = **0.9918**

### 5.3 Error Analysis & Bottleneck Findings
Analysis of the 500 assessment queries (1,734 true links) via `experiments/inspect_missed.py` revealed:
- **Blocking Misses (21 links, 1.21%):** Hard name truncations or phonetic variations in India records (e.g. `Sunrise Consulting Pvt Ltd` vs `Sunrise Consultancies`) where char n-gram overlap fell below the 0.10 threshold.
- **Classifier False Negatives (28 links):** Entities with matching names but conflicting street addresses due to branch relocations.
- **Classifier False Positives (8 links):** Co-located entities in the same industrial complex sharing similar commercial names.
- **Singleton Handling (0 False Positives):** Zero spurious links predicted for singleton queries due to the 0.75 decision threshold.

### 5.4 Throughput & Full Inference Feasibility
- **Full Target Index Size:** 10,320,219 records (India: 4.13M, US: 6.19M).
- **Index Build Time:** 628.5s (India) and 1055.1s (US).
- **Query Throughput:** 1.39s/query (India) and 1.83s/query (US) on single-core CPU sparse TF-IDF.
- **Full Test Set Scale:** 1,732,544 queries across India (810k), US (663k), and France (259k).
- **Inference Runtime Implication:** At ~1.5 seconds per query on a single CPU core, full sequential test set inference requires $\approx 721$ CPU hours ($\approx 30$ days). For hackathon deployment within standard resource windows, query batching, multithreading, or approximate vector search (e.g., HNSW/Faiss or PySpark) would be necessary for linear cluster scaling.

---

## 6. Conclusion
The repaired pipeline establishes honest, verified benchmarks strictly on organizer data:
1. Multi-strategy country-partitioned blocking achieves >94.8% recall on uncurated 10.3M record indexes.
2. Grouped HistGradientBoosting with threshold 0.75 achieves 0.9893 macro $F_{0.5}$ on held-out assessment.
3. Singleton entities are preserved with 100% accuracy.
4. Output writing is memory-bounded and atomically promoted, with strict isolation between sample smoke runs and submission outputs.

---

## Appendix: Additional Experiments

### Feature Ablation Study (Genuine Retraining & Threshold Tuning on Tune Split)
| Feature Set | Features | Optimal Threshold | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Singleton Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| A. Name features only | 15 | 0.70 | 0.9405 | 0.8373 | 0.9057 | 0.8571 |
| B. Address features only | 14 | 0.45 | 0.9567 | 0.9063 | 0.9383 | 0.8929 |
| C. Name + Address basic | 29 | 0.60 | 0.9968 | 0.9808 | 0.9920 | 1.0000 |
| D. Name + Address + Numeric (no meta) | 28 | 0.75 | 0.9954 | 0.9770 | 0.9897 | 0.9643 |
| **E. Full Features (All Signals)** | **31** | **0.75** | **0.9978** | **0.9785** | **0.9921** | **1.0000** |
