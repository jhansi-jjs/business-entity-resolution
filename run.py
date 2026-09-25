#!/usr/bin/env python3
"""Run script for Business Entity Resolution Challenge.

Executes the complete end-to-end inference pipeline and validates the final submission files.
Usage:
    python run.py [--sample SAMPLE_SIZE] [--no-validator]
"""

import sys
from pathlib import Path
import argparse

# Add repo root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Submission Pipeline")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional sample size for fast test execution (default: full test set)"
    )
    parser.add_argument(
        "--no-validator",
        action="store_true",
        help="Skip organizer validator step"
    )
    args = parser.parse_args()

    results = run_pipeline(
        sample_size=args.sample,
        run_validator=not args.no_validator
    )

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Validator Status: {results['validator_status']}")
    print(f"Matching Results: {results['matching_file']}")
    print(f"Candidate Pairs:  {results['candidate_file']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
