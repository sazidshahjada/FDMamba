"""Formula, reduction, gradient and precision regressions for multiclass losses."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from utils.losses import CrossEntropy, BalancedCrossEntropy, FocalLoss


torch.set_num_threads(2)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.2, -0.3, 0.5], [-0.7, 0.2, 1.3], [0.4, 0.8, -0.2]],
                                   dtype=torch.float64, requires_grad=True)
        self.targets = torch.tensor([0, 1, 2])

    def test_cross_entropy_matches_pytorch_values_and_gradients(self):
        for reduction in ('none', 'sum', 'mean'):
            for weight in (None, [0.5, 2., 3.]):
                for smoothing in (0., 0.1):
                    with self.subTest(reduction=reduction, weight=weight, smoothing=smoothing):
                        criterion = CrossEntropy(weight=weight, reduction=reduction, label_smoothing=smoothing)
                        result = criterion(self.logits, self.targets)
                        expected = F.cross_entropy(
                            self.logits, self.targets,
                            weight=None if weight is None else torch.tensor(weight, dtype=torch.float64),
                            reduction=reduction, label_smoothing=smoothing,
                        )
                        torch.testing.assert_close(result, expected)
                        actual_grad = torch.autograd.grad(result.sum(), self.logits, retain_graph=True)[0]
                        expected_grad = torch.autograd.grad(expected.sum(), self.logits, retain_graph=True)[0]
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_balanced_weights_dataset_order_and_buffers(self):
        dataset = SimpleNamespace(classes=['0', '1', '2'], targets=[0] * 8 + [1] * 3 + [2])
        criterion = BalancedCrossEntropy.from_dataset(dataset)
        torch.testing.assert_close(criterion.weight, torch.tensor([0.5, 4/3, 4.]))
        torch.testing.assert_close(criterion.class_counts, torch.tensor([8., 3., 1.], dtype=torch.float64))
        expected = F.cross_entropy(self.logits, self.targets, weight=criterion.weight.double())
        torch.testing.assert_close(criterion(self.logits, self.targets), expected)
        self.assertEqual(set(criterion.state_dict()), {'weight', 'class_counts'})
        restored = BalancedCrossEntropy([1, 1, 1])
        restored.load_state_dict(criterion.state_dict())
        torch.testing.assert_close(restored(self.logits, self.targets), expected)
        self.assertEqual(list(criterion.parameters()), [])
        large_counts = torch.tensor([100_000_001, 30_000_003, 10_000_001])
        torch.testing.assert_close(BalancedCrossEntropy(large_counts).class_counts, large_counts.double())

    def test_balanced_equal_counts_match_plain_cross_entropy(self):
        torch.testing.assert_close(BalancedCrossEntropy([8, 8, 8])(self.logits, self.targets),
                                   CrossEntropy()(self.logits, self.targets))

    def test_focal_matches_independent_softmax_formula(self):
        probabilities = self.logits.softmax(dim=1)[torch.arange(3), self.targets]
        for alpha in (None, 0.25, [0.5, 2., 3.]):
            alpha_tensor = torch.ones(3, dtype=torch.float64) if alpha is None else torch.tensor(alpha, dtype=torch.float64)
            expected = -alpha_tensor * (1 - probabilities).pow(2.) * probabilities.log()
            for reduction in ('none', 'sum', 'mean'):
                with self.subTest(alpha=alpha, reduction=reduction):
                    result = FocalLoss(gamma=2., alpha=alpha, reduction=reduction)(self.logits, self.targets)
                    reduced = expected if reduction == 'none' else getattr(expected, reduction)()
                    torch.testing.assert_close(result, reduced)
                    actual_grad = torch.autograd.grad(result.sum(), self.logits, retain_graph=True)[0]
                    expected_grad = torch.autograd.grad(reduced.sum(), self.logits, retain_graph=True)[0]
                    torch.testing.assert_close(actual_grad, expected_grad)

    def test_zero_gamma_matches_ce_and_alpha_is_linear(self):
        for reduction in ('none', 'sum', 'mean'):
            plain = CrossEntropy(reduction=reduction)(self.logits, self.targets)
            torch.testing.assert_close(FocalLoss(gamma=0, reduction=reduction)(self.logits, self.targets), plain)
            torch.testing.assert_close(FocalLoss(gamma=0, alpha=0.5, reduction=reduction)(self.logits, self.targets), 0.5 * plain)
        vector = torch.tensor([0.5, 2., 3.], dtype=torch.float64)
        expected = (F.cross_entropy(self.logits, self.targets, reduction='none') * vector).mean()
        torch.testing.assert_close(FocalLoss(gamma=0, alpha=vector)(self.logits, self.targets), expected)

    def test_ignored_labels_and_zero_effective_batches(self):
        for constructor, kwargs in ((CrossEntropy, {'weight': [0.5, 2., 3.], 'label_smoothing': 0.1}),
                                    (BalancedCrossEntropy, {'class_counts': [8, 3, 1]}),
                                    (FocalLoss, {'alpha': [0.5, 2., 3.]})):
            for reduction in ('none', 'sum', 'mean'):
                with self.subTest(loss=constructor.__name__, reduction=reduction):
                    criterion = constructor(**kwargs, reduction=reduction, ignore_index=-1)
                    targets = torch.tensor([0, -1, 2])
                    actual = criterion(self.logits, targets)
                    subset = criterion(self.logits[[0, 2]], self.targets[[0, 2]])
                    if reduction == 'none':
                        self.assertEqual(actual[1].item(), 0)
                        torch.testing.assert_close(actual[[0, 2]], subset)
                    else:
                        torch.testing.assert_close(actual, subset)
                    ignored = criterion(self.logits, torch.full((3,), -1, dtype=torch.long))
                    self.assertEqual(ignored.sum().item(), 0)
                    gradient = torch.autograd.grad(ignored.sum(), self.logits, retain_graph=True)[0]
                    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
                    empty = criterion(self.logits[:0], self.targets[:0])
                    self.assertEqual(empty.sum().item(), 0)
                    empty_gradient = torch.autograd.grad(empty.sum(), self.logits, retain_graph=True)[0]
                    torch.testing.assert_close(empty_gradient, torch.zeros_like(empty_gradient))
        criterion = CrossEntropy(weight=[0., 0., 0.])
        self.assertEqual(criterion(self.logits, self.targets).item(), 0)

    def test_extreme_logits_and_fractional_gamma_have_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for gamma in (0, 0.5, 2.):
                with self.subTest(dtype=dtype, gamma=gamma):
                    logits = torch.tensor([[1000., -1000., 0.], [1000., -1000., 0.]], dtype=dtype, requires_grad=True)
                    targets = torch.tensor([0, 1])
                    value = FocalLoss(gamma=gamma, alpha=[0.5, 2., 3.])(logits, targets)
                    self.assertTrue(torch.isfinite(value))
                    value.backward()
                    self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_gradcheck(self):
        self.assertTrue(torch.autograd.gradcheck(FocalLoss(gamma=1.5, alpha=[0.5, 2., 3.]),
                                                (self.logits, self.targets)))

    def test_cpu_autocast_and_optimizer_step_for_all_losses(self):
        for criterion in (CrossEntropy(), BalancedCrossEntropy([8, 3, 1]), FocalLoss(alpha=[0.5, 2., 3.])):
            with self.subTest(loss=type(criterion).__name__):
                model = nn.Linear(4, 3)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before = model.weight.detach().clone()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    value = criterion(model(torch.randn(3, 4)), self.targets)
                self.assertEqual(value.dtype, torch.float32)
                value.backward()
                self.assertTrue(torch.isfinite(model.weight.grad).all())
                optimizer.step()
                self.assertFalse(torch.equal(model.weight, before))

    def test_invalid_settings_and_input_shapes(self):
        for constructor, kwargs in (
            (BalancedCrossEntropy, {'class_counts': [1, 0, 2]}),
            (BalancedCrossEntropy, {'class_counts': []}),
            (BalancedCrossEntropy, {'class_counts': [1, float('nan'), 2]}),
            (CrossEntropy, {'weight': [-1, 1, 1]}),
            (CrossEntropy, {'label_smoothing': 1.5}),
            (FocalLoss, {'gamma': -1}), (FocalLoss, {'gamma': float('inf')}),
            (FocalLoss, {'alpha': -0.5}), (FocalLoss, {'alpha': [1, -1, 2]}),
            (FocalLoss, {'reduction': 'invalid'}),
        ):
            with self.subTest(loss=constructor.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                constructor(**kwargs)
        for criterion in (CrossEntropy(), FocalLoss(), BalancedCrossEntropy([1, 1, 1])):
            with self.subTest(loss=type(criterion).__name__):
                with self.assertRaises(TypeError):
                    criterion(self.logits, self.targets.float())
                with self.assertRaises(ValueError):
                    criterion(self.logits, self.targets[:, None])
                with self.assertRaises(ValueError):
                    criterion(self.logits[:, :, None], self.targets)
        for criterion in (CrossEntropy(weight=[1., 2.]), FocalLoss(alpha=[1., 2.])):
            with self.assertRaises(ValueError):
                criterion(self.logits, self.targets)
        with self.assertRaises(ValueError):
            BalancedCrossEntropy.from_dataset(SimpleNamespace(classes=['0', '1'], targets=[0, 0]))
        with self.assertRaises(TypeError):
            BalancedCrossEntropy.from_dataset(object())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA loss/AMP checks require a GPU')
    def test_cuda_amp_buffers_and_gradients(self):
        for criterion in (CrossEntropy(weight=[0.5, 2., 3.]), BalancedCrossEntropy([8, 3, 1]),
                          FocalLoss(alpha=[0.5, 2., 3.])):
            with self.subTest(loss=type(criterion).__name__):
                criterion = criterion.cuda()
                model = nn.Linear(4, 3).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scaler = torch.amp.GradScaler('cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    value = criterion(model(torch.randn(3, 4, device='cuda')), self.targets.cuda())
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                self.assertTrue(torch.isfinite(model.weight.grad).all())
                scaler.step(optimizer)
                scaler.update()
                self.assertTrue(all(buffer.is_cuda for buffer in criterion.buffers()))


if __name__ == '__main__':
    unittest.main()
