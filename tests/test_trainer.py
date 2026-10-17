"""Training/checkpoint/evaluation regressions using small deterministic datasets."""
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset

from fdmambadr.backbone import Backbone
from fdmambadr.fdconv_models import FDConvBackbone
from fdmambadr.hybrid_models import HybridBackbone
from fdmambadr.mamba_models import MambaBackbone
from fdmambadr.wrapper import BackboneWrapper
from utils.losses import CrossEntropy, BalancedCrossEntropy, FocalLoss
from utils.metrics import ClassificationMetrics
from utils.trainer import Trainer, seed_everything, warmup_cosine_scheduler


torch.set_num_threads(2)


class TinyClassifier(nn.Module):
    num_classes = 3

    def __init__(self, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(dropout), nn.Linear(8, 3))

    def forward(self, x):
        return self.layers(x)


class CountingDataset(Dataset):
    def __init__(self, inputs, labels, augment=False):
        self.inputs, self.targets = inputs, labels
        self.calls, self.augment = 0, augment

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        self.calls += 1
        image = self.inputs[index]
        if self.augment:
            image = image + (random.random() + np.random.rand() + torch.rand(())) * 0.05
        return image, self.targets[index]


class ScriptedValidationTrainer(Trainer):
    """Inject synthetic validation scores solely to exercise checkpoint policy."""
    scripted_scores = (0.1, 0.3, 0.2)

    def _run_epoch(self, loader, **kwargs):
        report, meter, indices = super()._run_epoch(loader, **kwargs)
        if kwargs['split'] == 'val':
            report['qwk'] = self.scripted_scores[min(kwargs['epoch'] - 1, len(self.scripted_scores) - 1)]
        return report, meter, indices


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        generator = torch.Generator().manual_seed(73)
        self.x = torch.randn(11, 4, generator=generator)
        self.y = torch.tensor([0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 0])
        seed_everything(42)
        self.initial = TinyClassifier(0.2).state_dict()

    def loaders(self, *, batch_size=3, augment=False):
        datasets = [CountingDataset(self.x.clone(), self.y.clone(), augment and i == 0) for i in range(3)]
        return [DataLoader(ds, batch_size=batch_size, shuffle=i == 0,
                           generator=torch.Generator().manual_seed(18 + i))
                for i, ds in enumerate(datasets)]

    def trainer(self, name, *, cls=Trainer, dropout=0.2, loaders=None, **kwargs):
        model = TinyClassifier(dropout)
        model.load_state_dict(self.initial)
        options = dict(epochs=3, output_dir=self.root / name, num_classes=3, verbose=False,
                       optimizer_kwargs={'lr': 0.01, 'weight_decay': 0.0}, patience=None)
        options.update(kwargs)
        return cls(model, *(loaders or self.loaders()), **options)

    def test_fit_artifacts_best_last_and_test_isolation(self):
        loaders = self.loaders()
        trainer = self.trainer('artifacts', cls=ScriptedValidationTrainer, loaders=loaders,
                               min_delta=0.5, accumulation_steps=2,
                               scheduler=torch.optim.lr_scheduler.StepLR,
                               scheduler_kwargs={'step_size': 1, 'gamma': 0.8})
        history = trainer.fit()
        self.assertEqual(len(history), 3)
        self.assertEqual(loaders[2].dataset.calls, 0)
        self.assertEqual(trainer.best_epoch, 2)  # min_delta does not hide strict best scores.
        self.assertEqual(trainer.global_step, 6)  # 4 minibatches -> 2 updates per epoch.
        for name in ('config.yaml', 'metadata.json', 'dataset_fingerprints.json', 'history.csv', 'history.json', 'best.pt', 'last.pt'):
            self.assertTrue((trainer.output_dir / name).is_file(), name)
        best = torch.load(trainer.output_dir / 'best.pt', weights_only=False)
        last = torch.load(trainer.output_dir / 'last.pt', weights_only=False)
        self.assertEqual(best['epoch'], 2)
        self.assertEqual(best['best_score'], 0.3)
        self.assertEqual(last['epoch'], 3)
        self.assertTrue(any(not torch.equal(best['model_state_dict'][key], value) for key, value in last['model_state_dict'].items()))
        state = copy.deepcopy(trainer.model.state_dict())
        rng = torch.get_rng_state().clone()
        loader_rng = loaders[2].generator.get_state().clone()
        trainer.model.train()
        report = trainer.test()
        self.assertEqual(report['num_samples'], len(self.y))
        self.assertTrue(report['checkpoint'].endswith('best.pt'))
        self.assertTrue(trainer.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_rng, loaders[2].generator.get_state()))
        for key, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        self.assertEqual(trainer.global_step, 6)
        with (trainer.output_dir / 'evaluations/test_predictions.csv').open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(self.y))
        self.assertEqual([int(row['true_label']) for row in rows], self.y.tolist())
        self.assertTrue(all(abs(sum(float(row[f'p{i}']) for i in range(3)) - 1) < 1e-6 for row in rows))
        json.dumps(report, allow_nan=False)

    def test_epoch_resume_matches_uninterrupted_training_with_all_rngs(self):
        options = dict(scheduler=torch.optim.lr_scheduler.StepLR,
                       scheduler_kwargs={'step_size': 1, 'gamma': 0.9}, accumulation_steps=2)
        uninterrupted = self.trainer('full', loaders=self.loaders(augment=True), **options)
        uninterrupted.fit(epochs=4)
        partial = self.trainer('resumed', loaders=self.loaders(augment=True), **options)
        partial.fit(epochs=2)
        resumed = self.trainer('resumed', loaders=self.loaders(augment=True), **options)
        resumed.fit(epochs=4, resume_from=partial.output_dir / 'last.pt')
        self.assertEqual(resumed.epoch, 4)
        self.assertEqual(resumed.global_step, uninterrupted.global_step)
        self.assertEqual(resumed.best_epoch, uninterrupted.best_epoch)
        for key, value in uninterrupted.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
        self.assertEqual(resumed.scheduler.state_dict(), uninterrupted.scheduler.state_dict())
        for a, b in zip(uninterrupted.history, resumed.history):
            self.assertEqual(a['val']['loss'], b['val']['loss'])
            self.assertEqual(a['learning_rates'], b['learning_rates'])
            self.assertEqual(a['val']['qwk'], b['val']['qwk'])

    def test_accumulation_matches_full_weighted_batch_and_loss_reductions(self):
        for criterion_class, options in (
            (CrossEntropy, {}), (BalancedCrossEntropy, {'class_counts': [8, 3, 1]}),
            (FocalLoss, {'alpha': [0.5, 2., 3.]}),
        ):
            for reduction in ('none', 'sum', 'mean'):
                with self.subTest(loss=criterion_class.__name__, reduction=reduction):
                    initial = nn.Linear(4, 3)
                    models = [copy.deepcopy(initial), copy.deepcopy(initial)]
                    trainers = []
                    for index, batch_size in enumerate((5, 2)):
                        loader = DataLoader(TensorDataset(self.x[:5], self.y[:5]), batch_size=batch_size)
                        trainer = Trainer(models[index], loader, num_classes=3,
                                          criterion=criterion_class(**options, reduction=reduction),
                                          optimizer=torch.optim.SGD, optimizer_kwargs={'lr': 0.1},
                                          accumulation_steps=1 if index == 0 else 3,
                                          output_dir=self.root / 'unused', verbose=False)
                        trainers.append(trainer)
                    full = trainers[0].train_one_epoch()
                    accumulated = trainers[1].train_one_epoch()
                    self.assertAlmostEqual(full['loss'], accumulated['loss'], places=6)
                    for a, b in zip(models[0].parameters(), models[1].parameters()):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
                    self.assertEqual(trainers[1].global_step, 1)

    def test_weighted_validation_loss_is_independent_of_batch_partition(self):
        criterion = CrossEntropy(weight=[0.2, 2., 3.], label_smoothing=0.1)
        trainer = self.trainer('weighted_eval', criterion=criterion, dropout=0.0)
        reports = [trainer.validate(DataLoader(TensorDataset(self.x, self.y), batch_size=size)) for size in (1, 4, 11)]
        for report in reports[1:]:
            self.assertAlmostEqual(report['loss'], reports[0]['loss'], places=6)

    def test_callbacks_early_stopping_and_undefined_monitor(self):
        events = []
        def callback(event, trainer, payload):
            events.append(event)
        trainer = self.trainer('early', cls=ScriptedValidationTrainer, epochs=10, patience=2,
                               callbacks=[callback])
        trainer.fit()
        self.assertEqual(trainer.epoch, 4)  # .1, .3, .2, .2: two nonimproving epochs.
        self.assertEqual(events[0], 'fit_start')
        self.assertEqual(events[-1], 'fit_end')
        trainer = self.trainer('undefined', cls=ScriptedValidationTrainer, patience=2)
        trainer.scripted_scores = (None,)
        trainer.fit(epochs=5)
        self.assertEqual(trainer.epoch, 2)
        self.assertFalse((trainer.output_dir / 'best.pt').exists())
        self.assertTrue((trainer.output_dir / 'last.pt').exists())
        with self.assertRaises(FileNotFoundError):
            trainer.test()
        trainer = self.trainer('undefined_error', cls=ScriptedValidationTrainer, undefined_monitor='error')
        trainer.scripted_scores = (None,)
        with self.assertRaisesRegex(ValueError, 'undefined'):
            trainer.fit()
        self.assertTrue((trainer.output_dir / 'last.pt').exists())

    def test_optimizer_scheduler_instances_hooks_and_cpu_amp(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)
        train, val, test = self.loaders()
        events = []
        trainer = Trainer(model, train, val, test, optimizer=optimizer, scheduler=scheduler,
                          criterion=FocalLoss(), amp=True, max_grad_norm=1.0,
                          monitor='loss', mode='min', epochs=2, output_dir=self.root / 'instances',
                          batch_parser=lambda batch: (batch[0], batch[1]),
                          loss_normalizer=lambda criterion, target: len(target),
                          metric_factory=lambda: ClassificationMetrics(3),
                          callbacks=[lambda event, owner, payload: events.append(event)], verbose=False)
        trainer.fit()
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertEqual(trainer.scheduler_interval, 'metric')
        trainer.test()
        self.assertIn('evaluation_end', events)

    def test_step_scheduler_and_warmup_cosine_factory(self):
        trainer = self.trainer('step', scheduler=torch.optim.lr_scheduler.StepLR,
                               scheduler_kwargs={'step_size': 1, 'gamma': 0.9}, scheduler_interval='step',
                               accumulation_steps=3)
        trainer.fit(epochs=2)
        self.assertEqual(trainer.scheduler.last_epoch, trainer.global_step)
        trainer = self.trainer('warmup', scheduler=warmup_cosine_scheduler,
                               scheduler_kwargs={'total_epochs': 3, 'warmup_epochs': 1})
        trainer.fit()
        rates = [entry['learning_rates'][0] for entry in trainer.history]
        self.assertLess(rates[0], rates[1])
        self.assertLess(rates[2], rates[1])

    def test_overwrite_resume_mismatch_and_eval_state_restoration_on_error(self):
        trainer = self.trainer('guards')
        trainer.fit(epochs=1)
        with self.assertRaises(FileExistsError):
            self.trainer('guards').fit()
        incompatible = self.trainer('other_arch', dropout=0.4)
        with self.assertRaisesRegex(ValueError, 'architecture'):
            incompatible.load_checkpoint(trainer.output_dir / 'last.pt')
        incompatible = self.trainer('other_data', loaders=self.loaders(batch_size=2))
        with self.assertRaisesRegex(ValueError, 'data/loader'):
            incompatible.load_checkpoint(trainer.output_dir / 'last.pt')
        bad_loader = DataLoader(TensorDataset(self.x, torch.full_like(self.y, 99)), batch_size=3)
        state = copy.deepcopy(trainer.model.state_dict())
        with self.assertRaises(ValueError):
            trainer.evaluate(bad_loader)
        for key, value in trainer.model.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))
        other_loss = self.trainer('metrics_only', criterion=FocalLoss())
        with self.assertRaises(ValueError):
            other_loss.evaluate(checkpoint=trainer.output_dir / 'best.pt')
        report = other_loss.evaluate(checkpoint=trainer.output_dir / 'best.pt', compute_loss=False, save=False)
        self.assertNotIn('loss', report)
        self.assertEqual(report['num_samples'], 11)

    def test_frozen_eval_only_model_and_saved_normalization_guard(self):
        loaders = self.loaders()
        for loader in loaders:
            loader.dataset.normalization_mean = (0.3, 0.4, 0.5)
            loader.dataset.normalization_std = (0.2, 0.2, 0.3)
        trainer = self.trainer('norm', loaders=loaders)
        trainer.fit(epochs=1)
        evaluation_model = TinyClassifier(0.2)
        for parameter in evaluation_model.parameters():
            parameter.requires_grad_(False)
        # Eval-only construction should work with a frozen model, too.
        evaluation_model.layers[-1].weight.requires_grad_(True)
        external = self.loaders()[2]
        evaluator = Trainer(evaluation_model, test_loader=external, num_classes=3,
                            output_dir=self.root / 'external', verbose=False)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            evaluator.evaluate(checkpoint=trainer.output_dir / 'best.pt')
        external.dataset.normalization_mean = (0.3, 0.4, 0.5)
        external.dataset.normalization_std = (0.2, 0.2, 0.3)
        report = evaluator.evaluate(checkpoint=trainer.output_dir / 'best.pt', split='external')
        self.assertEqual(report['num_samples'], 11)

    def test_bad_loaders_and_nonfinite_losses_are_rejected(self):
        loaders = self.loaders()
        with self.assertRaises(ValueError):
            self.trainer('same', loaders=[loaders[0], loaders[0], loaders[2]])
        with self.assertRaises(ValueError):
            self.trainer('shuffled_eval', loaders=[loaders[0], DataLoader(loaders[1].dataset, shuffle=True), loaders[2]])
        with self.assertRaises(ValueError):
            self.trainer('drop_eval', loaders=[loaders[0], DataLoader(loaders[1].dataset, batch_size=3, drop_last=True), loaders[2]])
        base = TensorDataset(self.x, self.y)
        overlapping = [DataLoader(Subset(base, indices), batch_size=2) for indices in ([0, 1, 2], [2, 3], [4, 5])]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.trainer('overlap', loaders=overlapping)
        trainer = self.trainer('nan')
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            trainer.fit(epochs=1)

    def test_all_baseline_and_fdconv_models_with_and_without_dense_wrapper(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(5, 3, 64, 64, generator=generator)
        y = torch.arange(5)
        for backbone in (Backbone, FDConvBackbone):
            for name in Backbone.supported_names():
                with self.subTest(backbone=backbone.__name__, name=name):
                    # Alternate native/custom heads across the architecture matrix.
                    model = BackboneWrapper(backbone, name=name, hidden_dims=(16,)) if name == 'mobilenetv3' else backbone(name)
                    loaders = [DataLoader(TensorDataset(x.clone(), y.clone()), batch_size=2) for _ in range(3)]
                    trainer = Trainer(model, *loaders, epochs=1, patience=None,
                                      output_dir=self.root / f'{backbone.__name__}_{name}', verbose=False)
                    trainer.fit()
                    self.assertEqual(trainer.test(save=False)['num_samples'], 5)

    @unittest.skipUnless(torch.cuda.is_available(), 'Real Mamba training requires CUDA')
    def test_mamba_and_hybrid_training_and_evaluation(self):
        for backbone in (MambaBackbone, HybridBackbone):
            for name in Backbone.supported_names():
                with self.subTest(backbone=backbone.__name__, name=name):
                    x = torch.randn(2, 3, 224, 224)
                    y = torch.tensor([0, 4])
                    loaders = [DataLoader(TensorDataset(x.clone(), y.clone()), batch_size=1) for _ in range(3)]
                    model = BackboneWrapper(backbone, name=name, hidden_dims=(16,), backbone_kwargs={'hidden_dim': 32})
                    trainer = Trainer(model, *loaders, device='cuda', epochs=1,
                                      output_dir=self.root / f'{backbone.__name__}_{name}', verbose=False)
                    trainer.fit()
                    self.assertEqual(trainer.test(save=False)['num_samples'], 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA AMP training requires CUDA')
    def test_cuda_amp_checkpoint_resume(self):
        trainer = self.trainer('gpu_amp', device='cuda', amp=True)
        trainer.fit(epochs=1)
        resumed = self.trainer('gpu_amp', device='cuda', amp=True)
        resumed.fit(epochs=2, resume_from=trainer.output_dir / 'last.pt')
        self.assertEqual(resumed.epoch, 2)
        self.assertTrue(resumed.scaler.is_enabled())


if __name__ == '__main__':
    unittest.main()
