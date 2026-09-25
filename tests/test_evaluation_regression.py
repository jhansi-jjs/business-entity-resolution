"""Regression tests for evaluation metrics, singleton rules, and split integrity."""

import sys
from pathlib import Path
import unittest
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_entity_metrics, evaluate_predictions


class TestEvaluationMetrics(unittest.TestCase):

    def test_singleton_perfect_credit(self):
        p, r, f05 = compute_entity_metrics(set(), set())
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertEqual(f05, 1.0)

    def test_singleton_false_positive(self):
        p, r, f05 = compute_entity_metrics(set(), {"S2-123"})
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f05, 0.0)

    def test_matching_entity_empty_prediction(self):
        p, r, f05 = compute_entity_metrics({"S2-100", "S3-200"}, set())
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)
        self.assertEqual(f05, 0.0)

    def test_manual_f05_calculation(self):
        p, r, f05 = compute_entity_metrics({"A", "B"}, {"A", "C"})
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 0.5)
        self.assertAlmostEqual(f05, 0.5)

    def test_manual_f05_precision_weighting(self):
        p, r, f05 = compute_entity_metrics({"A"}, {"A", "B", "C", "D", "E"})
        expected_f05 = (1.25 * 0.2 * 1.0) / (0.25 * 0.2 + 1.0)
        self.assertAlmostEqual(f05, expected_f05, places=5)

    def test_no_singletons_reports_na(self):
        gt = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        pred = {"S1-1": {"S2-1"}, "S1-2": {"S2-2"}}
        res = evaluate_predictions(pred, gt, ["S1-1", "S1-2"])
        self.assertEqual(res["singleton_count"], 0)
        self.assertEqual(res["singleton_accuracy"], "N/A")
        self.assertIsNone(res["singleton_accuracy_raw"])

    def test_zero_candidate_queries_retained(self):
        gt = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-3"}}
        pred = {"S1-1": {"S2-1"}, "S1-2": set()}
        res = evaluate_predictions(pred, gt, ["S1-1", "S1-2", "S1-3"])
        self.assertEqual(res["total_evaluated_s1"], 3)
        self.assertAlmostEqual(res["macro_f0_5_raw"], 2.0 / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
