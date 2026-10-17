"""Configurable single-device classification training, resume and evaluation.

Works with B0/B1/B2/B3 and BackboneWrapper. Training never reads test batches.
Checkpoints are selected on validation only; test() evaluates best.pt by default.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SequentialSampler, Subset, TensorDataset
import yaml

from .losses import CrossEntropy
from .metrics import ClassificationMetrics


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Call BEFORE constructing models/loaders to seed their initialization too."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def warmup_cosine_scheduler(optimizer: Optimizer, *, total_epochs: int,
                            warmup_epochs: int = 5, min_lr: float = 0.0,
                            start_factor: float = 0.1):
    """Epoch scheduler factory usable as Trainer(scheduler=..., scheduler_kwargs=...)."""
    if not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs)")
    if not 0 < start_factor <= 1 or not math.isfinite(min_lr) or min_lr < 0:
        raise ValueError("start_factor must be in (0,1] and min_lr must be finite/nonnegative")
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr,
    )
    if not warmup_epochs:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def _name(value) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}" if hasattr(value, '__qualname__') else _name(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _sample(dataset, index):
    if isinstance(dataset, Subset):
        return _sample(dataset.dataset, int(dataset.indices[index]))
    if hasattr(dataset, 'samples'):
        path, target = dataset.samples[index]
        return str(path), int(target)
    if hasattr(dataset, 'targets'):
        return '', int(dataset.targets[index])
    if isinstance(dataset, TensorDataset) and dataset.tensors[-1].ndim == 1:
        return '', int(dataset.tensors[-1][index])
    return None


def _loader_info(loader) -> dict | None:
    if loader is None:
        return None
    dataset = loader.dataset
    base = _base_dataset(dataset)
    root = Path(base.root).expanduser().resolve() if hasattr(base, 'root') else None
    digest = hashlib.sha256()
    known = True
    for index in range(len(dataset)):
        sample = _sample(dataset, index)
        if sample is None:
            known = False
            break
        path, label = sample
        if path and root is not None:
            path = os.path.relpath(path, root)
        digest.update(json.dumps([index, path, label], ensure_ascii=False).encode() + b'\n')
    return _jsonable(dict(
        dataset_class=_name(dataset), num_samples=len(dataset),
        root=root, class_to_idx=getattr(base, 'class_to_idx', None),
        classes=getattr(base, 'classes', None),
        manifest_sha256=digest.hexdigest() if known else None,
        identity_scope='ordered paths/labels only; image bytes not hashed' if known else 'dataset type/length only',
        transform=repr(getattr(base, 'transform', None)),
        normalization_mean=getattr(base, 'normalization_mean', None),
        normalization_std=getattr(base, 'normalization_std', None),
        batch_size=loader.batch_size, drop_last=loader.drop_last,
        num_workers=loader.num_workers, sampler_class=_name(loader.sampler),
    ))


def _loader_rng(loader):
    if loader is None:
        return {}
    result = {}
    for name, owner in (('loader', loader), ('sampler', loader.sampler)):
        generator = getattr(owner, 'generator', None)
        if isinstance(generator, torch.Generator):
            result[name] = generator.get_state()
    return result


def _restore_loader_rng(loader, state):
    if loader is None:
        return
    for name, owner in (('loader', loader), ('sampler', loader.sampler)):
        if name in state:
            generator = getattr(owner, 'generator', None)
            if not isinstance(generator, torch.Generator):
                raise ValueError(f"Resuming requires the saved {name} generator")
            generator.set_state(state[name].cpu())


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def _check_eval_loader(loader) -> None:
    if loader is None:
        raise ValueError("An evaluation DataLoader is required")
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Evaluation requires a sequential loader with drop_last=False")
    if len(loader.dataset) == 0 or len(loader) == 0:
        raise ValueError("Evaluation dataset/loader is empty")
    def check(transform):
        if transform is None:
            return
        if type(transform).__name__.startswith('Random') or type(transform).__name__ == 'ColorJitter':
            raise ValueError("Evaluation transforms must be deterministic; remove random augmentation")
        for child in getattr(transform, 'transforms', ()):
            check(child)
    check(getattr(_base_dataset(loader.dataset), 'transform', None))


class Trainer:
    """Single-device trainer for any project classifier returning raw [B,C] logits.

    Optimizer/scheduler accept instances OR factories/classes with *_kwargs.
    Scheduler interval is 'epoch', 'step' (optimizer updates), or 'metric'; auto
    selects metric for ReduceLROnPlateau, epoch otherwise. monitor is a validation
    report key (including 'loss' or dotted keys); default QWK/max. Ties keep the
    earlier checkpoint. min_delta affects early stopping, not best selection.

    Losses may return mean/sum scalars or per-example values. They are normalized
    to a mean objective, including across incomplete accumulation windows. The
    denominator is the CE target-weight sum or otherwise the nonignored count;
    custom criteria can supply loss_normalizer(criterion, targets).

    callbacks receive (event, trainer, payload) for fit_start, batch_end,
    epoch_end, fit_end and evaluation_end. Set trainer.should_stop=True to stop
    after an epoch. A custom metric_factory must provide update()/compute() and,
    if exporting predictions, get_predictions(), following ClassificationMetrics.

    Pass already constructed models. The trainer seeds run randomness; call
    seed_everything BEFORE model construction to seed model initialization.
    Epoch-boundary resume restores optimizer/scheduler/scaler/loss and RNG state.
    Exact augmentation replay is tested with num_workers=0; worker process RNGs
    of persistent multiprocessing loaders cannot be checkpointed here.
    """

    def __init__(
        self, model: nn.Module, train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None, test_loader: DataLoader | None = None, *,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | Callable = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Any = None, scheduler_kwargs: Mapping[str, Any] | None = None,
        scheduler_interval: str = 'auto', scheduler_monitor: str | None = None,
        epochs: int = 100, device: str | torch.device | None = None,
        amp: bool = False, amp_dtype: torch.dtype | None = None,
        accumulation_steps: int = 1, max_grad_norm: float | None = None,
        monitor: str = 'qwk', mode: str = 'max', patience: int | None = 15,
        min_delta: float = 0.0, undefined_monitor: str = 'skip',
        output_dir: str | Path = 'experiments/run',
        num_classes: int | None = None, class_names: Sequence[str] | None = None,
        seed: int = 42, deterministic: bool = False,
        metrics_kwargs: Mapping[str, Any] | None = None,
        metric_factory: Callable | None = None,
        batch_parser: Callable | None = None, loss_normalizer: Callable | None = None,
        callbacks: Sequence[Callable] = (), tensorboard: bool = False,
        save_predictions: bool = True, verbose: bool = True,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        for name, value in (('epochs', epochs), ('accumulation_steps', accumulation_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if patience is not None and (not isinstance(patience, int) or isinstance(patience, bool) or patience < 1):
            raise ValueError("patience must be a positive integer or None")
        if mode not in ('min', 'max') or undefined_monitor not in ('skip', 'error'):
            raise ValueError("mode must be min/max; undefined_monitor must be skip/error")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a constructed nn.Module")
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("callbacks must be callable")
        self.device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError("This trainer supports CPU or CUDA devices")
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype or (torch.float16 if self.device.type == 'cuda' else torch.bfloat16)
        if self.amp_dtype not in (torch.float16, torch.bfloat16) or (self.amp and self.device.type == 'cpu' and self.amp_dtype != torch.bfloat16):
            raise ValueError("AMP requires float16/bfloat16 on CUDA, or bfloat16 on CPU")
        self.seed, self.deterministic = seed, deterministic
        seed_everything(seed, deterministic)
        self.model = model.to(self.device)
        self.criterion = (criterion if criterion is not None else CrossEntropy()).to(self.device)
        self.loaders = dict(train=train_loader, val=val_loader, test=test_loader)
        for loader in (val_loader, test_loader):
            if loader is not None:
                _check_eval_loader(loader)
        datasets = [loader.dataset for loader in self.loaders.values() if loader is not None]
        if len({id(dataset) for dataset in datasets}) != len(datasets):
            raise ValueError("Use separate training/validation/test datasets")
        reference = next((_base_dataset(ds) for ds in datasets if hasattr(_base_dataset(ds), 'classes')), None)
        self.num_classes = num_classes if num_classes is not None else getattr(model, 'num_classes', None)
        if self.num_classes is None:
            self.num_classes = len(reference.classes) if reference is not None else 5
        self.class_names = tuple(class_names) if class_names is not None else tuple(reference.classes) if reference is not None else tuple(str(i) for i in range(self.num_classes))
        self.ignore_index = getattr(self.criterion, 'ignore_index', -100)
        self.metrics_kwargs = dict(metrics_kwargs or {})
        for reserved in ('num_classes', 'class_names', 'from_logits', 'ignore_index'):
            if reserved in self.metrics_kwargs:
                raise ValueError(f"Set {reserved} using the trainer/model/loss configuration")
        self.metrics_kwargs.setdefault('referable_grade', 2 if self.num_classes > 2 else 1)
        ClassificationMetrics(self.num_classes, class_names=self.class_names, ignore_index=self.ignore_index, **self.metrics_kwargs)
        for ds in datasets:
            classes = getattr(_base_dataset(ds), 'classes', None)
            if classes is not None and tuple(classes) != self.class_names:
                raise ValueError("All split class names/order must match the trainer")
        self.dataset_info = {key: _loader_info(loader) for key, loader in self.loaders.items()}
        roots = [item['root'] for item in self.dataset_info.values() if item and item['root']]
        if len(set(roots)) != len(roots) and not all(isinstance(ds, Subset) for ds in datasets):
            raise ValueError("Split roots must differ; use explicit disjoint Subsets for a shared dataset")
        for index, root in enumerate(roots):
            for other in roots[index + 1:]:
                if root != other and (Path(root) in Path(other).parents or Path(other) in Path(root).parents):
                    raise ValueError("Training/validation/test roots must not be nested")
        # Shared bases are valid only when Subset indices are disjoint.
        def lineage(dataset):
            indices = list(range(len(dataset)))
            while isinstance(dataset, Subset):
                indices = [int(dataset.indices[i]) for i in indices]
                dataset = dataset.dataset
            return dataset, set(indices)
        lineages = [lineage(ds) for ds in datasets]
        for index, (base, indices) in enumerate(lineages):
            if any(base is other and indices.intersection(other_indices)
                   for other, other_indices in lineages[index + 1:]):
                raise ValueError("Training/validation/test Subsets must not overlap")
        self.metric_factory, self.batch_parser = metric_factory, batch_parser
        self.loss_normalizer = loss_normalizer
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("The model has no trainable parameters")
        options = dict(optimizer_kwargs or {})
        if isinstance(optimizer, Optimizer):
            if options:
                raise ValueError("optimizer_kwargs cannot accompany an optimizer instance")
            self.optimizer = optimizer
        else:
            if optimizer is torch.optim.AdamW:
                options = {'lr': 1e-3, 'weight_decay': 1e-4, **options}
            self.optimizer = optimizer(parameters, **options)
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must produce a torch Optimizer")
        model_ids = {id(p) for p in self.model.parameters()}
        if any(id(p) not in model_ids for group in self.optimizer.param_groups for p in group['params']):
            raise ValueError("Optimizer parameters must belong to this model; construct it for the selected device")
        self.monitor, self.mode = monitor, mode
        self.scheduler_monitor = scheduler_monitor or monitor
        scheduler_options = dict(scheduler_kwargs or {})
        self.scheduler_spec = dict(factory=_jsonable(scheduler) if isinstance(scheduler, type) or callable(scheduler) else _name(scheduler) if scheduler is not None else None,
                                   kwargs=_jsonable(scheduler_options))
        if scheduler is None and scheduler_options:
            raise ValueError("scheduler_kwargs requires a scheduler")
        if scheduler is not None and not hasattr(scheduler, 'optimizer'):
            if isinstance(scheduler, type) and issubclass(scheduler, ReduceLROnPlateau):
                scheduler_options.setdefault('mode', mode)
            scheduler = scheduler(self.optimizer, **scheduler_options)
        elif scheduler_options:
            raise ValueError("scheduler_kwargs cannot accompany a scheduler instance")
        if scheduler is not None and scheduler.optimizer is not self.optimizer:
            raise ValueError("Scheduler must use this trainer's optimizer")
        self.scheduler = scheduler
        self.scheduler_interval = ('metric' if isinstance(scheduler, ReduceLROnPlateau) else 'epoch') if scheduler_interval == 'auto' else scheduler_interval
        if self.scheduler_interval not in ('epoch', 'step', 'metric'):
            raise ValueError("scheduler_interval must be auto, epoch, step or metric")
        if isinstance(scheduler, ReduceLROnPlateau) and self.scheduler_interval != 'metric':
            raise ValueError("ReduceLROnPlateau requires scheduler_interval='metric'")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp and self.device.type == 'cuda' and self.amp_dtype == torch.float16)
        self.max_epochs, self.accumulation_steps = epochs, accumulation_steps
        self.max_grad_norm, self.patience, self.min_delta = max_grad_norm, patience, float(min_delta)
        self.undefined_monitor = undefined_monitor
        self.output_dir = Path(output_dir).expanduser()
        self.callbacks, self.tensorboard = tuple(callbacks), tensorboard
        self.save_predictions, self.verbose = save_predictions, verbose
        self.run_config = dict(run_config or {})
        self.epoch = self.global_step = self.bad_epochs = 0
        self.best_score = self.best_val_qwk = self.early_stop_score = None
        self.best_epoch = None
        self.history = []
        self.should_stop = False
        self._started = False
        self._writer = None

    def _new_metrics(self):
        if self.metric_factory is not None:
            return self.metric_factory()
        return ClassificationMetrics(self.num_classes, class_names=self.class_names,
                                     ignore_index=self.ignore_index, **self.metrics_kwargs)

    def _event(self, event, **payload):
        for callback in self.callbacks:
            callback(event, self, payload)

    def _batch(self, batch):
        if self.batch_parser is not None:
            images, targets = self.batch_parser(batch)
        elif isinstance(batch, Mapping):
            images = batch['images'] if 'images' in batch else batch['image']
            targets = batch['labels'] if 'labels' in batch else batch['label']
        else:
            images, targets = batch[:2]
        if not isinstance(images, Tensor) or not images.is_floating_point():
            raise TypeError("Batches must contain floating-point input tensors")
        if not isinstance(targets, Tensor) or targets.dtype != torch.long or targets.ndim != 1:
            raise TypeError("Batch labels must be one-dimensional torch.long tensors")
        if images.shape[0] != len(targets):
            raise ValueError("Input/label batch sizes differ")
        valid = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        if ((targets[valid] < 0) | (targets[valid] >= self.num_classes)).any():
            raise ValueError("Batch labels are outside the configured class range")
        return images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    def _loss_terms(self, logits, targets):
        valid = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
        if self.loss_normalizer is not None:
            denominator = torch.as_tensor(self.loss_normalizer(self.criterion, targets), device=self.device, dtype=torch.float32)
        elif isinstance(self.criterion, nn.CrossEntropyLoss) and self.criterion.weight is not None:
            denominator = self.criterion.weight[targets[valid]].float().sum()
        else:
            denominator = valid.sum().float()
        if denominator.ndim != 0 or not torch.isfinite(denominator) or denominator < 0:
            raise ValueError("Loss normalizer must be a finite nonnegative scalar")
        if denominator.item() == 0:
            return logits.float().sum() * 0, 0.0
        loss = self.criterion(logits, targets)
        if not isinstance(loss, Tensor) or not torch.isfinite(loss).all():
            raise FloatingPointError("Criterion returned a nonfinite/non-tensor loss")
        reduction = getattr(self.criterion, 'reduction', 'mean')
        if loss.ndim == 1 and loss.shape == targets.shape:
            numerator = loss[valid].sum()
        elif loss.ndim == 0 and reduction == 'sum':
            numerator = loss
        elif loss.ndim == 0 and reduction == 'mean':
            numerator = loss * denominator
        else:
            raise ValueError("Criterion must return a mean/sum scalar or one loss per example")
        if not torch.isfinite(numerator):
            raise FloatingPointError("Loss numerator is nonfinite")
        return numerator, float(denominator.item())

    def _optimizer_step(self, denominator):
        if denominator <= 0:
            self.model.zero_grad(set_to_none=True)
            return
        self.scaler.unscale_(self.optimizer)
        optimized = [p for group in self.optimizer.param_groups for p in group['params'] if p.grad is not None]
        if not optimized:
            raise RuntimeError("Loss produced no gradients for optimizer parameters")
        for parameter in optimized:
            parameter.grad.div_(denominator)
        norm = nn.utils.clip_grad_norm_(optimized, self.max_grad_norm or float('inf'),
                                       error_if_nonfinite=not self.scaler.is_enabled())
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        stepped = self.scaler.get_scale() >= old_scale
        if stepped and not math.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite gradient norm")
        if stepped:
            self.global_step += 1
            if self.scheduler is not None and self.scheduler_interval == 'step':
                self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

    def _run_epoch(self, loader, *, training, split, epoch, compute_loss=True):
        if loader is None or len(loader) == 0:
            raise ValueError(f"{split} loader is empty or missing")
        if not training:
            _check_eval_loader(loader)
        meter = self._new_metrics()
        total_loss = total_weight = window_weight = 0.0
        sample_indices, offset = [], 0
        started = time.perf_counter()
        model_mode, criterion_mode = self.model.training, self.criterion.training
        self.model.train(training)
        self.criterion.train(training)
        if training:
            self.model.zero_grad(set_to_none=True)
            if callable(getattr(loader.sampler, 'set_epoch', None)):
                loader.sampler.set_epoch(epoch)
        try:
            with torch.enable_grad() if training else torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    images, targets = self._batch(batch)
                    with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                        logits = self.model(images)
                        if not isinstance(logits, Tensor) or logits.shape != (len(targets), self.num_classes):
                            raise ValueError(f"Model must return logits [B,{self.num_classes}]")
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError("Model returned nonfinite logits")
                        numerator, weight = self._loss_terms(logits, targets) if compute_loss else (None, 0.0)
                    if compute_loss:
                        total_loss += float(numerator.detach())
                        total_weight += weight
                    if training and weight > 0:
                        self.scaler.scale(numerator).backward()
                        window_weight += weight
                    if training and ((batch_index + 1) % self.accumulation_steps == 0 or batch_index + 1 == len(loader)):
                        self._optimizer_step(window_weight)
                        window_weight = 0.0
                    meter.update(logits, targets)
                    valid = targets != self.ignore_index if self.ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
                    sample_indices.extend((offset + valid.nonzero().flatten().cpu().numpy()).tolist())
                    offset += len(targets)
                    self._event('batch_end', split=split, epoch=epoch, batch=batch_index,
                                global_step=self.global_step,
                                loss=float(numerator.detach()) / weight if weight else None)
            report = meter.compute()
            if compute_loss:
                if total_weight <= 0:
                    raise ValueError(f"{split} has no examples with positive loss weight")
                report['loss'] = total_loss / total_weight
            report['seconds'] = time.perf_counter() - started
            return report, meter, sample_indices
        finally:
            self.model.train(model_mode)
            self.criterion.train(criterion_mode)

    def train_one_epoch(self, epoch: int | None = None) -> dict:
        """Run one training epoch; fit() owns epoch-level scheduler/checkpoint bookkeeping."""
        return self._run_epoch(self.loaders['train'], training=True, split='train',
                               epoch=epoch if epoch is not None else self.epoch + 1)[0]

    def validate(self, loader: DataLoader | None = None) -> dict:
        """Evaluate current weights on validation, without changing training state."""
        return self._run_epoch(loader if loader is not None else self.loaders['val'],
                               training=False, split='val', epoch=self.epoch)[0]

    @staticmethod
    def _value(report, key):
        value = report
        for part in key.split('.'):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(f"Metric {key!r} is absent from the validation report")
            value = value[part]
        if value is None:
            return None
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Monitored metric {key!r} must be finite or None")
        return float(value)

    def _better(self, value, previous, delta=0.0):
        return previous is None or (value > previous + delta if self.mode == 'max' else value < previous - delta)

    def _criterion_settings(self):
        return dict(class_name=_name(self.criterion), **{
            key: getattr(self.criterion, key) for key in ('reduction', 'ignore_index', 'label_smoothing', 'gamma')
            if hasattr(self.criterion, key)
        }, buffer_shapes={key: list(value.shape) for key, value in self.criterion.state_dict().items()})

    def _model_signature(self):
        data = _jsonable(dict(class_name=_name(self.model), structure=repr(self.model),
                              config=getattr(self.model, 'config', None),
                              state={key: [list(value.shape), str(value.dtype)] for key, value in self.model.state_dict().items()}))
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _config(self):
        modules = []
        for path, module in self.model.named_modules():
            if hasattr(module, 'replacement'):
                modules.append(dict(path=path, fdconv=module.replacement))
        return _jsonable(dict(
            model_class=_name(self.model), model_config=getattr(self.model, 'config', None),
            model_structure=repr(self.model), resolved_fdconv=modules,
            criterion=self._criterion_settings(), loss_buffers=self.criterion.state_dict(),
            optimizer_class=_name(self.optimizer), optimizer_defaults=self.optimizer.defaults,
            scheduler_class=_name(self.scheduler) if self.scheduler is not None else None,
            scheduler_spec=self.scheduler_spec,
            scheduler_interval=self.scheduler_interval, scheduler_monitor=self.scheduler_monitor,
            epochs=self.max_epochs, accumulation_steps=self.accumulation_steps,
            max_grad_norm=self.max_grad_norm, monitor=self.monitor, mode=self.mode,
            patience=self.patience, min_delta=self.min_delta, tie_policy='first strict optimum',
            undefined_monitor=self.undefined_monitor, seed=self.seed, deterministic=self.deterministic,
            device=self.device, amp=self.amp, amp_dtype=self.amp_dtype,
            num_classes=self.num_classes, class_names=self.class_names,
            metrics_kwargs=self.metrics_kwargs, run_config=self.run_config,
            custom_hooks=dict(batch_parser=self.batch_parser, loss_normalizer=self.loss_normalizer,
                              metric_factory=self.metric_factory),
            trainable={key: parameter.requires_grad for key, parameter in self.model.named_parameters()},
        ))

    def _prepare_output(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'config.yaml').write_text(yaml.safe_dump(self._config(), sort_keys=False))
        _write_json(self.output_dir / 'dataset_fingerprints.json', self.dataset_info)
        versions = {}
        for package in ('torch', 'torchvision', 'timm', 'mamba-ssm', 'causal-conv1d', 'numpy', 'scikit-learn'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        _write_json(self.output_dir / 'metadata.json', dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
            platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
            packages=versions, device=str(self.device), torch_cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            parameters=sum(p.numel() for p in self.model.parameters()),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            resume_scope='epoch boundary; exact worker augmentation replay requires num_workers=0',
        ))
        if self.tensorboard and self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.output_dir / 'tensorboard'), purge_step=self.epoch + 1)

    def _save_history(self):
        _write_json(self.output_dir / 'history.json', self.history)
        rows = []
        for entry in self.history:
            row = {key: entry[key] for key in ('epoch', 'global_step', 'monitored_value', 'improved', 'selection_status')}
            row['learning_rates'] = json.dumps(entry['learning_rates'])
            for split in ('train', 'val'):
                for key, value in entry[split].items():
                    if value is None or isinstance(value, (int, float)):
                        row[f'{split}_{key}'] = value
            rows.append(row)
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.output_dir / 'history.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save a complete epoch-boundary checkpoint atomically."""
        path = Path(path) if path is not None else self.output_dir / 'last.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(
            format_version=1, model_state_dict=self.model.state_dict(), model_signature=self._model_signature(),
            optimizer_state_dict=self.optimizer.state_dict(), optimizer_class=_name(self.optimizer),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,
            scheduler_class=_name(self.scheduler) if self.scheduler is not None else None,
            criterion_state_dict=self.criterion.state_dict(), criterion_settings=self._criterion_settings(),
            scaler_state_dict=self.scaler.state_dict(), epoch=self.epoch, global_step=self.global_step,
            best_score=self.best_score, best_val_qwk=self.best_val_qwk, best_epoch=self.best_epoch,
            early_stop_score=self.early_stop_score, bad_epochs=self.bad_epochs,
            history=self.history, seed=self.seed, config=self._config(), class_names=self.class_names,
            dataset_fingerprints=self.dataset_info, rng_state=_rng_state(),
            loader_rng={key: _loader_rng(loader) for key, loader in self.loaders.items()},
        )
        temporary = path.with_name(path.name + '.tmp')
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    def _read_checkpoint(self, checkpoint):
        path = self.output_dir / f'{checkpoint}.pt' if str(checkpoint) in ('best', 'last') else Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        # Full local training checkpoints include Python/NumPy RNG state.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format_version') != 1 or state['model_signature'] != self._model_signature():
            raise ValueError("Checkpoint model architecture/configuration does not match")
        if tuple(state['class_names']) != self.class_names:
            raise ValueError("Checkpoint class order does not match")
        return state, path

    def load_checkpoint(self, checkpoint='last', *, resume: bool = True) -> dict:
        """Load this trainer's local checkpoint; resume=True also restores run/RNG state."""
        state, _path = self._read_checkpoint(checkpoint)
        if state['criterion_settings'] != self._criterion_settings():
            raise ValueError("Checkpoint loss type/settings differ; construct the matching criterion")
        if resume:
            current = self._config()
            for key in ('monitor', 'mode', 'min_delta', 'accumulation_steps', 'max_grad_norm',
                        'amp', 'amp_dtype', 'metrics_kwargs', 'scheduler_interval', 'scheduler_monitor',
                        'scheduler_spec', 'optimizer_defaults', 'custom_hooks', 'trainable', 'deterministic'):
                if state['config'][key] != current[key]:
                    raise ValueError(f"Resume setting differs: {key}")
            if state['optimizer_class'] != _name(self.optimizer) or state['scheduler_class'] != (_name(self.scheduler) if self.scheduler is not None else None):
                raise ValueError("Checkpoint optimizer/scheduler types differ")
            for split in ('train', 'val'):
                old, new = state['dataset_fingerprints'][split], self.dataset_info[split]
                if old != new:
                    raise ValueError(f"Resume {split} data/loader metadata differs")
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.criterion.load_state_dict(state['criterion_state_dict'], strict=True)
        if resume:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            self.scaler.load_state_dict(state['scaler_state_dict'])
            for key in ('epoch', 'global_step', 'best_score', 'best_val_qwk', 'best_epoch', 'early_stop_score', 'bad_epochs', 'seed'):
                setattr(self, key, state[key])
            self.history = copy.deepcopy(state['history'])
            for split, loader in self.loaders.items():
                _restore_loader_rng(loader, state['loader_rng'][split])
            _restore_rng(state['rng_state'])
            self._started = True
        return state

    def fit(self, epochs: int | None = None, *, resume_from: str | Path | None = None) -> list[dict]:
        """Train up to a TOTAL epoch count, optionally resuming last.pt (not extra epochs)."""
        if self.loaders['train'] is None or self.loaders['val'] is None:
            raise ValueError("fit requires training and validation loaders")
        target_epochs = self.max_epochs if epochs is None else epochs
        if not isinstance(target_epochs, int) or isinstance(target_epochs, bool) or target_epochs < 1:
            raise ValueError("epochs must be a positive total count")
        if resume_from is not None:
            self.load_checkpoint(resume_from, resume=True)
        if not self._started and any((self.output_dir / file).exists() for file in ('last.pt', 'best.pt', 'history.json', 'config.yaml')):
            raise FileExistsError("Run directory already contains a run; use resume_from or a new output_dir")
        self.max_epochs = target_epochs
        self._prepare_output()
        self._started, self.should_stop = True, False
        self._event('fit_start', start_epoch=self.epoch + 1, total_epochs=target_epochs)
        try:
            for epoch in range(self.epoch + 1, target_epochs + 1):
                learning_rates = [group['lr'] for group in self.optimizer.param_groups]
                train, _train_meter, _indices = self._run_epoch(self.loaders['train'], training=True, split='train', epoch=epoch)
                val, meter, indices = self._run_epoch(self.loaders['val'], training=False, split='val', epoch=epoch)
                score = self._value(val, self.monitor)
                improved = score is not None and self._better(score, self.best_score)
                if improved:
                    self.best_score, self.best_epoch = score, epoch
                if score is not None and self._better(score, self.early_stop_score, self.min_delta):
                    self.early_stop_score, self.bad_epochs = score, 0
                else:
                    self.bad_epochs += 1
                qwk = val.get('qwk')
                if qwk is not None:
                    self.best_val_qwk = qwk if self.best_val_qwk is None else max(qwk, self.best_val_qwk)
                if self.scheduler is not None:
                    if self.scheduler_interval == 'epoch':
                        self.scheduler.step()
                    elif self.scheduler_interval == 'metric':
                        scheduler_value = self._value(val, self.scheduler_monitor)
                        if scheduler_value is not None:
                            self.scheduler.step(scheduler_value)
                self.epoch = epoch
                record = dict(epoch=epoch, train=train, val=val, learning_rates=learning_rates,
                              global_step=self.global_step, monitored_value=score, improved=improved,
                              selection_status='defined' if score is not None else 'undefined')
                self.history.append(record)
                if self._writer is not None:
                    for split, report in (('train', train), ('val', val)):
                        for key, value in report.items():
                            if isinstance(value, (int, float)):
                                self._writer.add_scalar(f'{split}/{key}', value, epoch)
                    self._writer.flush()
                self._event('epoch_end', **record)
                self._save_history()
                self.save_checkpoint()
                if improved:
                    self.save_checkpoint(self.output_dir / 'best.pt')
                    self._export('val_best', val, meter, indices, self.loaders['val'])
                if self.verbose:
                    value = f'{score:.6f}' if score is not None else 'undefined'
                    print(f"Epoch {epoch}/{target_epochs} train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} val_{self.monitor}={value}", flush=True)
                if score is None and self.undefined_monitor == 'error':
                    raise ValueError(f"Validation {self.monitor} is undefined; last.pt retained, no undefined best score saved")
                if self.should_stop or self.patience is not None and self.bad_epochs >= self.patience:
                    break
            self._event('fit_end', epoch=self.epoch, best_epoch=self.best_epoch, best_score=self.best_score)
            return self.history
        finally:
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    def _export(self, split, report, meter, indices, loader):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', split):
            raise ValueError("split must be a simple name containing letters, numbers, '_' or '-'")
        directory = self.output_dir / 'evaluations'
        _write_json(directory / f'{split}_metrics.json', report)
        if not self.save_predictions:
            return
        data = meter.get_predictions()
        if data['probabilities'] is None:
            raise ValueError("Prediction export requires class probability matrices")
        base = _base_dataset(loader.dataset)
        root = getattr(base, 'root', None)
        with (directory / f'{split}_predictions.csv').open('w', newline='') as file:
            fields = ['sample_index', 'sample_id', 'image_path', 'true_label', 'predicted_label'] + [f'p{i}' for i in range(self.num_classes)]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row, index in enumerate(indices):
                sample = _sample(loader.dataset, index)
                path = sample[0] if sample is not None else ''
                if path and root is not None:
                    path = os.path.relpath(path, root)
                record = dict(sample_index=index, sample_id=path or str(index), image_path=path,
                              true_label=int(data['targets'][row]), predicted_label=int(data['predictions'][row]))
                record.update({f'p{i}': float(data['probabilities'][row, i]) for i in range(self.num_classes)})
                writer.writerow(record)

    def evaluate(self, loader: DataLoader | None = None, *, split: str = 'test',
                 checkpoint: str | Path | None = 'best', save: bool = True,
                 compute_loss: bool = True) -> dict:
        """Evaluate a frozen checkpoint on test/external data without optimizer updates.

        None evaluates current weights. Requested checkpoint weights/loss buffers
        are temporary; current model, loss, modes and RNGs are restored afterwards.
        Pass compute_loss=False for metrics-only evaluation with a different loss
        configuration. Sequential, deterministic loaders retain all examples.
        """
        loader = loader if loader is not None else self.loaders['test']
        _check_eval_loader(loader)
        classes = getattr(_base_dataset(loader.dataset), 'classes', None)
        if classes is not None and tuple(classes) != self.class_names:
            raise ValueError("Evaluation label order differs from training")
        base = _base_dataset(loader.dataset)
        train_info = self.dataset_info.get('train')
        def check_normalization(train_info):
            if train_info is None:
                return
            for field in ('normalization_mean', 'normalization_std'):
                if train_info[field] != _jsonable(getattr(base, field, None)):
                    raise ValueError("Evaluation normalization must match the saved training normalization")
        check_normalization(train_info)
        rng, loader_rng = _rng_state(), _loader_rng(loader)
        model_state = copy.deepcopy(self.model.state_dict()) if checkpoint is not None else None
        loss_state = copy.deepcopy(self.criterion.state_dict()) if checkpoint is not None and compute_loss else None
        try:
            checkpoint_path = None
            if checkpoint is not None:
                state, checkpoint_path = self._read_checkpoint(checkpoint)
                check_normalization(state['dataset_fingerprints'].get('train'))
                if compute_loss and state['criterion_settings'] != self._criterion_settings():
                    raise ValueError("Evaluation loss differs from checkpoint; supply the matching criterion or compute_loss=False")
                self.model.load_state_dict(state['model_state_dict'], strict=True)
                if compute_loss:
                    self.criterion.load_state_dict(state['criterion_state_dict'], strict=True)
            report, meter, indices = self._run_epoch(loader, training=False, split=split, epoch=self.epoch,
                                                     compute_loss=compute_loss)
            report['checkpoint'] = str(checkpoint_path) if checkpoint_path is not None else None
            report['split'] = split
            if save:
                self._export(split, report, meter, indices, loader)
            self._event('evaluation_end', split=split, report=report)
            return report
        finally:
            if model_state is not None:
                self.model.load_state_dict(model_state, strict=True)
            if loss_state is not None:
                self.criterion.load_state_dict(loss_state, strict=True)
            _restore_loader_rng(loader, loader_rng)
            _restore_rng(rng)

    def test(self, *, checkpoint: str | Path | None = 'best', save: bool = True,
             compute_loss: bool = True) -> dict:
        """Evaluate the configured test loader (best validation checkpoint by default)."""
        return self.evaluate(split='test', checkpoint=checkpoint, save=save, compute_loss=compute_loss)
