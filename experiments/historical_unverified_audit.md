# Historical Unverified Artifacts Audit Note

## Background
In commit `51aa9df` of this repository, several experiment reports and CSVs were committed:
- `experiments/ablation_results.csv`
- `experiments/experiments.csv`
- Corresponding claims in earlier documentation drafts

## Verification Findings

1. **Ablation Studies Script (`experiments/ablation_studies.py`)**:
   - Inspection of lines 49-55 of `experiments/ablation_studies.py` revealed that the values written to `ablation_results.csv` were hardcoded Python dictionaries rather than actual runtime evaluation outputs from model retraining and evaluation.
   - No feature masking, retraining, or threshold tuning was executed during the generation of `ablation_results.csv`.

2. **Singleton Accuracy Artifact**:
   - `experiments.csv` reported `singleton_accuracy = 1.0000` across all models.
   - Historical evaluation logic previously defaulted to `1.0` when zero singletons were present in the evaluated sample slice, rather than reporting `None` or `"N/A"`.
   - Sample selection in `run_training_experiments.py` was drawn from the top of `train_source1.tsv` without stratified sampling across singleton and multi-match cardinality classes.

3. **Distractor Pool Scaling Bug**:
   - In `experiments/run_training_experiments.py`, distractor collection checked `len(s2_distractors) * 250000 < distractor_cap`. Since `len(s2_distractors)` counts chunks (1 chunk = 1), after the first chunk `1 * 250000 >= 15000`, prematurely halting distractor collection.

## Corrective Action
- The historical CSV files are categorized as unverified artifacts from commit `51aa9df`.
- All subsequent benchmarks, ablation studies, and evaluation tables in this repository are strictly derived from executed Python scripts running on real splits (`train`, `tune`, and held-out `assessment`) with real measurements and exact metric formulas.
- Exact unrounded floating-point metrics will be computed and preserved.
