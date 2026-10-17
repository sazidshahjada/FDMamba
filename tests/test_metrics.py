"""Reference comparisons and degenerate/split-aggregation metric regressions."""
import json
import unittest

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from utils.metrics import ClassificationMetrics, compute_metrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.array([0, 1, 2, 3, 4, 4, 2, 0, 3, 1])
        self.predictions = np.array([0, 2, 2, 4, 4, 1, 3, 1, 3, 0])
        self.probabilities = np.full((10, 5), 0.05)
        self.probabilities[np.arange(10), self.predictions] = 0.8

    def test_classification_metrics_match_sklearn(self):
        result = compute_metrics(self.predictions, self.targets)
        self.assertAlmostEqual(result['accuracy'], accuracy_score(self.targets, self.predictions))
        self.assertAlmostEqual(result['balanced_accuracy'], balanced_accuracy_score(self.targets, self.predictions))
        self.assertAlmostEqual(result['macro_f1'], f1_score(self.targets, self.predictions, average='macro', labels=range(5), zero_division=0))
        self.assertAlmostEqual(result['qwk'], cohen_kappa_score(self.targets, self.predictions, labels=range(5), weights='quadratic'))
        np.testing.assert_array_equal(result['confusion_matrix'], confusion_matrix(self.targets, self.predictions, labels=range(5)))
        precision, recall, f1, support = precision_recall_fscore_support(self.targets, self.predictions, labels=range(5), zero_division=0)
        for i, row in enumerate(result['per_class']):
            self.assertEqual(row['label'], i)
            self.assertAlmostEqual(row['precision'], precision[i])
            self.assertAlmostEqual(row['recall'], recall[i])
            self.assertAlmostEqual(row['f1'], f1[i])
            self.assertEqual(row['support'], support[i])
        json.dumps(result, allow_nan=False)

    def test_logits_probabilities_and_epoch_accumulation_agree(self):
        expected = compute_metrics(self.probabilities, self.targets, from_logits=False)
        logits = torch.tensor(np.log(self.probabilities), requires_grad=True)
        self.assertEqual(compute_metrics(logits, torch.tensor(self.targets)), expected)
        metrics = ClassificationMetrics()
        # Unequal batches must contribute by sample, never by averaging batch metrics.
        metrics.update(logits[:2], self.targets[:2])
        metrics.update(logits[2:9], self.targets[2:9])
        metrics.update(logits[9:], self.targets[9:])
        self.assertEqual(metrics.compute(), expected)
        self.assertIsNone(logits.grad)
        saved = metrics.get_predictions()
        np.testing.assert_array_equal(saved['targets'], self.targets)
        np.testing.assert_allclose(saved['probabilities'], self.probabilities)
        saved['targets'][0] = 4
        np.testing.assert_array_equal(metrics.get_predictions()['targets'], self.targets)
        metrics.reset()
        with self.assertRaises(ValueError):
            metrics.compute()

    def test_ordinal_distances_include_absent_intermediate_grades(self):
        targets = np.array([0, 0, 2, 4, 4, 4])
        predictions = np.array([0, 4, 0, 2, 4, 4])
        result = compute_metrics(predictions, targets)
        self.assertAlmostEqual(result['qwk'], cohen_kappa_score(targets, predictions, labels=range(5), weights='quadratic'))
        self.assertEqual(len(result['confusion_matrix']), 5)

    def test_absent_classes_and_undefined_scores_are_explicit(self):
        result = compute_metrics([0, 4, 0, 4], [0, 4, 0, 4])
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['balanced_accuracy'], 1)
        self.assertEqual(result['macro_f1'], 0.4)
        self.assertEqual(result['per_class'][2]['support'], 0)
        self.assertEqual(result['qwk'], 1)
        single = compute_metrics([0, 0], [0, 0])
        self.assertIsNone(single['qwk'])
        self.assertIn('qwk', single['undefined_metrics'])
        self.assertIsNone(single['referable_dr']['sensitivity'])
        self.assertIsNone(single['referable_dr']['roc_auc'])
        self.assertEqual(single['referable_dr']['specificity'], 1)
        probabilities = np.zeros((2, 5))
        probabilities[:, 4] = 1
        positive = compute_metrics(probabilities, [4, 4], from_logits=False)
        self.assertIsNone(positive['referable_dr']['specificity'])
        self.assertIsNone(positive['referable_dr']['roc_auc'])
        self.assertEqual(positive['referable_dr']['sensitivity'], 1)
        json.dumps(positive, allow_nan=False)

    def test_referable_dr_uses_summed_probabilities(self):
        # First example's grade argmax is 0, but total referable probability > .5.
        probabilities = np.array([[0.35, 0.1, 0.2, 0.2, 0.15],
                                  [0.6, 0.1, 0.1, 0.1, 0.1],
                                  [0.1, 0.1, 0.6, 0.1, 0.1],
                                  [0.2, 0.4, 0.1, 0.1, 0.2]])
        targets = np.array([2, 0, 4, 1])
        result = compute_metrics(probabilities, targets, from_logits=False)['referable_dr']
        self.assertEqual(result['sensitivity'], 1)
        self.assertEqual(result['specificity'], 1)
        self.assertAlmostEqual(result['roc_auc'], roc_auc_score(targets >= 2, probabilities[:, 2:].sum(axis=1)))
        self.assertEqual(result['confusion_matrix'], [[2, 0], [0, 2]])
        hard = compute_metrics(probabilities.argmax(axis=1), targets)['referable_dr']
        self.assertEqual(hard['sensitivity'], 0.5)
        self.assertIsNone(hard['roc_auc'])
        self.assertEqual(hard['decision_rule'], 'predicted_grade')

    def test_ignored_and_empty_batches(self):
        metrics = ClassificationMetrics(from_logits=False)
        metrics.update(np.empty((0, 5)), [])
        with self.assertRaises(ValueError):
            metrics.compute()
        targets = self.targets.copy()
        targets[0] = -100
        metrics.update(self.probabilities, targets)
        expected = compute_metrics(self.probabilities[1:], targets[1:], from_logits=False)
        self.assertEqual(metrics.compute(), expected)
        metrics.update(np.ones((1, 5)) / 5, [-100])
        self.assertEqual(metrics.compute(), expected)

    def test_configuration_and_invalid_inputs(self):
        for options in ({'num_classes': 1}, {'class_names': ['DR']}, {'class_names': ['0'] * 5},
                        {'from_logits': 'False'}, {'referable_grade': 5}, {'referable_threshold': float('nan')}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                ClassificationMetrics(**options)
        for predictions, targets, options in (
            ([0, 1], [0], {}), ([0], [5], {}), ([-1], [0], {}),
            ([1.5], [0], {}), ([0], [0.5], {}),
            ([[float('nan')] * 5], [0], {}), ([[float('inf')] * 5], [0], {}),
            ([[1., 0.]], [0], {}),
        ):
            with self.subTest(predictions=predictions, targets=targets), self.assertRaises((ValueError, TypeError)):
                compute_metrics(predictions, targets, **options)
        with self.assertRaises(ValueError):
            compute_metrics([[0.3] * 5], [0], from_logits=False)
        with self.assertRaises(ValueError):
            compute_metrics([[1.2, -0.2, 0, 0, 0]], [0], from_logits=False)
        metrics = ClassificationMetrics()
        metrics.update([0], [0])
        with self.assertRaises(ValueError):
            metrics.update([[1., 0., 0., 0., 0.]], [0])

    def test_custom_classes_binary_mode_and_extreme_logits(self):
        binary = compute_metrics([0, 1], [0, 1], num_classes=2, referable_grade=1, class_names=['no', 'yes'])
        self.assertEqual(binary['qwk'], 1)
        self.assertEqual(binary['class_names'], ['no', 'yes'])
        disabled = compute_metrics([0], [0], referable_grade=None)
        self.assertNotIn('referable_dr', disabled)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logits = torch.tensor([[1000., -1000., 0., 0., 0.], [0., 0., 0., -1000., 1000.]], dtype=dtype)
            result = compute_metrics(logits, [0, 4])
            self.assertEqual(result['accuracy'], 1)
            json.dumps(result, allow_nan=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA tensor ingestion requires a GPU')
    def test_cuda_logits_are_detached_to_cpu(self):
        logits = torch.tensor(np.log(self.probabilities), device='cuda', requires_grad=True)
        targets = torch.tensor(self.targets, device='cuda')
        expected = compute_metrics(self.probabilities, self.targets, from_logits=False)
        self.assertEqual(compute_metrics(logits, targets), expected)
        self.assertIsNone(logits.grad)


if __name__ == '__main__':
    unittest.main()
