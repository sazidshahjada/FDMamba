"""Configurable single-device classification training, resume and evaluation.

Works with B0/B1/B2/B3 and BackboneWrapper. Checkpoints are selected on validation
only; fit() evaluates the supplied test loader once after the training loop.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from .data_checks import _base_dataset, _check_eval_loader, _class_counts, _loader_info, _sample
from .engine import EpochRunner
from .losses import CrossEntropy
from .metrics import ClassificationMetrics
from .runtime import (
    _jsonable,
    _loader_rng,
    _name,
    _restore_loader_rng,
    _restore_rng,
    _rng_state,
    _write_json,
    seed_everything,
)
from .runtime import (
    warmup_cosine_scheduler as warmup_cosine_scheduler,
)


class Trainer(EpochRunner):
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

    Completed fits export descriptive best/last checkpoint copies to
    saved_weights_dir; set it to None to disable these copies. verbose controls
    run metadata, split summaries, progress bars and completion messages.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        *,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | Callable = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        scheduler: Any = None,
        scheduler_kwargs: Mapping[str, Any] | None = None,
        scheduler_interval: str = "auto",
        scheduler_monitor: str | None = None,
        epochs: int = 100,
        device: str | torch.device | None = None,
        amp: bool = False,
        amp_dtype: torch.dtype | None = None,
        accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        monitor: str = "qwk",
        mode: str = "max",
        patience: int | None = 15,
        min_delta: float = 0.0,
        undefined_monitor: str = "skip",
        output_dir: str | Path = "experiments/run",
        saved_weights_dir: str | Path | None = "saved_weights",
        num_classes: int | None = None,
        class_names: Sequence[str] | None = None,
        seed: int = 42,
        deterministic: bool = False,
        metrics_kwargs: Mapping[str, Any] | None = None,
        metric_factory: Callable | None = None,
        batch_parser: Callable | None = None,
        loss_normalizer: Callable | None = None,
        callbacks: Sequence[Callable] = (),
        tensorboard: bool = False,
        save_predictions: bool = True,
        verbose: bool = True,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        for name, value in (("epochs", epochs), ("accumulation_steps", accumulation_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if patience is not None and (
            not isinstance(patience, int) or isinstance(patience, bool) or patience < 1
        ):
            raise ValueError("patience must be a positive integer or None")
        if mode not in ("min", "max") or undefined_monitor not in ("skip", "error"):
            raise ValueError("mode must be min/max; undefined_monitor must be skip/error")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a constructed nn.Module")
        if any(not callable(callback) for callback in callbacks):
            raise TypeError("callbacks must be callable")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("This trainer supports CPU or CUDA devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype or (
            torch.float16 if self.device.type == "cuda" else torch.bfloat16
        )
        if self.amp_dtype not in (torch.float16, torch.bfloat16) or (
            self.amp and self.device.type == "cpu" and self.amp_dtype != torch.bfloat16
        ):
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
        reference = next(
            (_base_dataset(ds) for ds in datasets if hasattr(_base_dataset(ds), "classes")), None
        )
        self.num_classes = (
            num_classes if num_classes is not None else getattr(model, "num_classes", None)
        )
        if self.num_classes is None:
            self.num_classes = len(reference.classes) if reference is not None else 5
        self.class_names = (
            tuple(class_names)
            if class_names is not None
            else tuple(reference.classes)
            if reference is not None
            else tuple(str(i) for i in range(self.num_classes))
        )
        self.ignore_index = getattr(self.criterion, "ignore_index", -100)
        self.metrics_kwargs = dict(metrics_kwargs or {})
        for reserved in ("num_classes", "class_names", "from_logits", "ignore_index"):
            if reserved in self.metrics_kwargs:
                raise ValueError(f"Set {reserved} using the trainer/model/loss configuration")
        self.metrics_kwargs.setdefault("referable_grade", 2 if self.num_classes > 2 else 1)
        ClassificationMetrics(
            self.num_classes,
            class_names=self.class_names,
            ignore_index=self.ignore_index,
            **self.metrics_kwargs,
        )
        for ds in datasets:
            classes = getattr(_base_dataset(ds), "classes", None)
            if classes is not None and tuple(classes) != self.class_names:
                raise ValueError("All split class names/order must match the trainer")
        self.dataset_info = {key: _loader_info(loader) for key, loader in self.loaders.items()}
        training_info = self.dataset_info["train"]
        if training_info is not None:
            for split in ("val", "test"):
                info = self.dataset_info[split]
                if info is not None:
                    for field in ("normalization_mean", "normalization_std"):
                        if info[field] != training_info[field]:
                            raise ValueError(
                                f"{split} normalization must match training normalization"
                            )
        roots = [item["root"] for item in self.dataset_info.values() if item and item["root"]]
        if len(set(roots)) != len(roots) and not all(isinstance(ds, Subset) for ds in datasets):
            raise ValueError(
                "Split roots must differ; use explicit disjoint Subsets for a shared dataset"
            )
        for index, root in enumerate(roots):
            for other in roots[index + 1 :]:
                if root != other and (
                    Path(root) in Path(other).parents or Path(other) in Path(root).parents
                ):
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
            if any(
                base is other and indices.intersection(other_indices)
                for other, other_indices in lineages[index + 1 :]
            ):
                raise ValueError("Training/validation/test Subsets must not overlap")
        self.metric_factory, self.batch_parser = metric_factory, batch_parser
        self.loss_normalizer = loss_normalizer
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters and train_loader is not None:
            raise ValueError("The training model has no trainable parameters")
        options = dict(optimizer_kwargs or {})
        if not parameters:
            if scheduler is not None:
                raise ValueError("An evaluation-only frozen model cannot use a scheduler")
            self.optimizer = None
        elif isinstance(optimizer, Optimizer):
            if options:
                raise ValueError("optimizer_kwargs cannot accompany an optimizer instance")
            self.optimizer = optimizer
        else:
            if optimizer is torch.optim.AdamW:
                options = {"lr": 1e-3, "weight_decay": 1e-4, **options}
            self.optimizer = optimizer(parameters, **options)
        if self.optimizer is not None and not isinstance(self.optimizer, Optimizer):
            raise TypeError("optimizer must produce a torch Optimizer")
        model_ids = {id(p) for p in self.model.parameters()}
        if any(
            id(p) not in model_ids
            for group in (self.optimizer.param_groups if self.optimizer else [])
            for p in group["params"]
        ):
            raise ValueError(
                "Optimizer parameters must belong to this model; construct it for the selected device"
            )
        self.monitor, self.mode = monitor, mode
        self.scheduler_monitor = scheduler_monitor or monitor
        scheduler_options = dict(scheduler_kwargs or {})
        self.scheduler_spec = dict(
            factory=_jsonable(scheduler)
            if isinstance(scheduler, type) or callable(scheduler)
            else _name(scheduler)
            if scheduler is not None
            else None,
            kwargs=_jsonable(scheduler_options),
        )
        if scheduler is None and scheduler_options:
            raise ValueError("scheduler_kwargs requires a scheduler")
        if scheduler is not None and not hasattr(scheduler, "optimizer"):
            if isinstance(scheduler, type) and issubclass(scheduler, ReduceLROnPlateau):
                scheduler_options.setdefault("mode", mode)
            scheduler = scheduler(self.optimizer, **scheduler_options)
        elif scheduler_options:
            raise ValueError("scheduler_kwargs cannot accompany a scheduler instance")
        if scheduler is not None and scheduler.optimizer is not self.optimizer:
            raise ValueError("Scheduler must use this trainer's optimizer")
        self.scheduler = scheduler
        self.scheduler_interval = (
            ("metric" if isinstance(scheduler, ReduceLROnPlateau) else "epoch")
            if scheduler_interval == "auto"
            else scheduler_interval
        )
        if self.scheduler_interval not in ("epoch", "step", "metric"):
            raise ValueError("scheduler_interval must be auto, epoch, step or metric")
        if isinstance(scheduler, ReduceLROnPlateau) and self.scheduler_interval != "metric":
            raise ValueError("ReduceLROnPlateau requires scheduler_interval='metric'")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp and self.device.type == "cuda" and self.amp_dtype == torch.float16,
        )
        self.max_epochs, self.accumulation_steps = epochs, accumulation_steps
        self.max_grad_norm, self.patience, self.min_delta = (
            max_grad_norm,
            patience,
            float(min_delta),
        )
        self.undefined_monitor = undefined_monitor
        self.output_dir = Path(output_dir).expanduser()
        self.saved_weights_dir = (
            Path(saved_weights_dir).expanduser() if saved_weights_dir is not None else None
        )
        self.saved_weights = {}
        self.callbacks, self.tensorboard = tuple(callbacks), tensorboard
        self.save_predictions, self.verbose = save_predictions, verbose
        self.run_config = dict(run_config or {})
        self.epoch = self.global_step = self.bad_epochs = 0
        self.best_score = self.best_val_qwk = self.early_stop_score = None
        self.best_epoch = None
        self.history = []
        self.test_report = None
        self.should_stop = False
        self._started = False
        self._writer = None

    def _new_metrics(self):
        """Create a fresh accumulator for one complete split."""
        if self.metric_factory is not None:
            return self.metric_factory()
        return ClassificationMetrics(
            self.num_classes,
            class_names=self.class_names,
            ignore_index=self.ignore_index,
            **self.metrics_kwargs,
        )

    def _event(self, event, **payload):
        """Notify optional callbacks of a training or evaluation event."""
        for callback in self.callbacks:
            callback(event, self, payload)

    def _model_name(self):
        """Describe the backbone and enabled method in logs and exported filenames."""
        build = getattr(self.model, "build_config", {})
        backbone = build.get("backbone", type(self.model).__name__)
        aliases = {
            "mobilenetv3": "mobilenetv3_large_100",
            "mobilenetv3_large": "mobilenetv3_large_100",
            "efficientnet": "efficientnet_b0",
            "resnet": "resnet50",
            "resnet_50": "resnet50",
        }
        backbone = aliases.get(backbone.lower().replace("-", "_"), backbone)
        methods = {"B0": "Baseline", "B1": "FDConv", "B2": "Mamba", "B3": "FDConv_Mamba"}
        variant = build.get("variant")
        return f"{backbone}_{variant}_{methods[variant]}" if variant in methods else backbone

    def _print_run_summary(self, stage, *, loaders=None, checkpoint=None):
        """Print device and split metadata once per run without reading any images."""
        if not self.verbose:
            return
        device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        )
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad
        )
        print(f"\n{stage}", flush=True)
        print(f"Device: {self.device} ({device_name})", flush=True)
        print(f"Model: {self._model_name()}", flush=True)
        print(f"Parameters: {total:,} total; {trainable:,} trainable", flush=True)
        print(f"Classes ({self.num_classes}): {', '.join(self.class_names)}", flush=True)
        precision = f"enabled ({self.amp_dtype})" if self.amp else "disabled"
        print(f"AMP: {precision}", flush=True)
        if checkpoint is not None:
            print(f"Checkpoint: {checkpoint}", flush=True)
        data = self.run_config.get("data", {})
        if "image_size" in data:
            print(f"Image size: {data['image_size']}", flush=True)
        for split, loader in (self.loaders if loaders is None else loaders).items():
            if loader is None:
                print(f"  {split}: not configured", flush=True)
                continue
            base = _base_dataset(loader.dataset)
            print(
                f"  {split}: {len(loader.dataset):,} samples, {len(loader):,} batches; "
                f"batch_size={loader.batch_size}, workers={loader.num_workers}, "
                f"drop_last={loader.drop_last}",
                flush=True,
            )
            if hasattr(base, "root"):
                print(f"    Path: {base.root}", flush=True)
            counts = _class_counts(loader.dataset, self.class_names)
            print(
                f"    Class counts: {counts if counts is not None else 'unavailable'}", flush=True
            )
        print(f"Output directory: {self.output_dir.resolve()}", flush=True)

    def _export_saved_weights(self):
        """Copy selected and final checkpoints atomically, retaining evaluation metadata."""
        if self.saved_weights_dir is None:
            return
        self.saved_weights_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._model_name()).strip("._-") or "model"
        run = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.output_dir.resolve().name)
        for kind, epoch in (("best", self.best_epoch), ("last", self.epoch)):
            if epoch is None:
                continue
            source = self.output_dir / f"{kind}.pt"
            destination = self.saved_weights_dir / (
                f"{name}_{run}_seed{self.seed}_epoch{epoch:04d}_{kind}.pt"
            )
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
            self.saved_weights[kind] = destination
            if self.verbose:
                print(f"Saved {kind} model: {destination.resolve()}", flush=True)

    def train_one_epoch(self, epoch: int | None = None) -> dict:
        """Run one training epoch; fit() owns epoch-level scheduler/checkpoint bookkeeping."""
        return self._run_epoch(
            self.loaders["train"],
            training=True,
            split="train",
            epoch=epoch if epoch is not None else self.epoch + 1,
        )[0]

    def validate(self, loader: DataLoader | None = None) -> dict:
        """Evaluate current weights on validation, without changing training state."""
        return self._run_epoch(
            loader if loader is not None else self.loaders["val"],
            training=False,
            split="val",
            epoch=self.epoch,
        )[0]

    @staticmethod
    def _value(report, key):
        """Resolve a finite scalar metric or an explicitly undefined value."""
        value = report
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(f"Metric {key!r} is absent from the validation report")
            value = value[part]
        if value is None:
            return None
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Monitored metric {key!r} must be finite or None")
        return float(value)

    def _better(self, value, previous, delta=0.0):
        """Compare a candidate score under the configured min/max direction."""
        return previous is None or (
            value > previous + delta if self.mode == "max" else value < previous - delta
        )

    @staticmethod
    def _format_metric(value, spec=".5f"):
        """Format optional report values without failing on undefined metrics."""
        return format(value, spec) if value is not None else "undefined"

    def _criterion_settings(self):
        """Describe loss behavior and buffer shapes for checkpoint compatibility."""
        return dict(
            class_name=_name(self.criterion),
            **{
                key: getattr(self.criterion, key)
                for key in ("reduction", "ignore_index", "label_smoothing", "gamma")
                if hasattr(self.criterion, key)
            },
            buffer_shapes={
                key: list(value.shape) for key, value in self.criterion.state_dict().items()
            },
        )

    def _model_signature(self):
        """Hash model structure, configuration and state shapes to reject mismatches."""
        data = _jsonable(
            dict(
                class_name=_name(self.model),
                structure=repr(self.model),
                config=getattr(self.model, "config", None),
                state={
                    key: [list(value.shape), str(value.dtype)]
                    for key, value in self.model.state_dict().items()
                },
            )
        )
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _config(self):
        """Collect the resolved run settings, including the selected FDConv location."""
        modules = []
        for path, module in self.model.named_modules():
            if hasattr(module, "replacement"):
                modules.append(dict(path=path, fdconv=module.replacement))
        return _jsonable(
            dict(
                model_class=_name(self.model),
                model_config=getattr(self.model, "config", None),
                model_structure=repr(self.model),
                resolved_fdconv=modules,
                criterion=self._criterion_settings(),
                loss_buffers=self.criterion.state_dict(),
                optimizer_class=_name(self.optimizer) if self.optimizer else None,
                optimizer_defaults=self.optimizer.defaults if self.optimizer else {},
                scheduler_class=_name(self.scheduler) if self.scheduler is not None else None,
                scheduler_spec=self.scheduler_spec,
                scheduler_interval=self.scheduler_interval,
                scheduler_monitor=self.scheduler_monitor,
                epochs=self.max_epochs,
                accumulation_steps=self.accumulation_steps,
                max_grad_norm=self.max_grad_norm,
                monitor=self.monitor,
                mode=self.mode,
                patience=self.patience,
                min_delta=self.min_delta,
                tie_policy="first strict optimum",
                undefined_monitor=self.undefined_monitor,
                seed=self.seed,
                deterministic=self.deterministic,
                device=self.device,
                amp=self.amp,
                amp_dtype=self.amp_dtype,
                num_classes=self.num_classes,
                class_names=self.class_names,
                metrics_kwargs=self.metrics_kwargs,
                run_config=self.run_config,
                saved_weights_dir=self.saved_weights_dir,
                model_build=getattr(self.model, "build_config", None),
                custom_hooks=dict(
                    batch_parser=self.batch_parser,
                    loss_normalizer=self.loss_normalizer,
                    metric_factory=self.metric_factory,
                ),
                trainable={
                    key: parameter.requires_grad for key, parameter in self.model.named_parameters()
                },
            )
        )

    def _prepare_output(self):
        """Save configuration, dataset identities and actual environment details."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.yaml").write_text(
            yaml.safe_dump(self._config(), sort_keys=False)
        )
        _write_json(self.output_dir / "dataset_fingerprints.json", self.dataset_info)
        versions = {}
        for package in (
            "torch",
            "torchvision",
            "timm",
            "mamba-ssm",
            "causal-conv1d",
            "numpy",
            "scikit-learn",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        _write_json(
            self.output_dir / "metadata.json",
            dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(),
                platform=platform.platform(),
                processor=platform.processor(),
                cpu_count=os.cpu_count(),
                packages=versions,
                device=str(self.device),
                torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                parameters=sum(p.numel() for p in self.model.parameters()),
                trainable_parameters=sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                resume_scope="epoch boundary; exact worker augmentation replay requires num_workers=0",
            ),
        )
        if self.tensorboard and self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(
                str(self.output_dir / "tensorboard"), purge_step=self.epoch + 1
            )

    def _save_history(self):
        """Write structured and tabular histories from completed epochs."""
        _write_json(self.output_dir / "history.json", self.history)
        rows = []
        for entry in self.history:
            row = {
                key: entry[key]
                for key in (
                    "epoch",
                    "global_step",
                    "monitored_value",
                    "improved",
                    "selection_status",
                )
            }
            row["learning_rates"] = json.dumps(entry["learning_rates"])
            for split in ("train", "val"):
                for key, value in entry[split].items():
                    if value is None or isinstance(value, (int, float)):
                        row[f"{split}_{key}"] = value
            rows.append(row)
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (self.output_dir / "history.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Save a complete epoch-boundary checkpoint atomically."""
        if self.optimizer is None:
            raise ValueError("Training checkpoints require an optimizer")
        path = Path(path) if path is not None else self.output_dir / "last.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(
            format_version=1,
            model_state_dict=self.model.state_dict(),
            model_signature=self._model_signature(),
            optimizer_state_dict=self.optimizer.state_dict(),
            optimizer_class=_name(self.optimizer),
            scheduler_state_dict=self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            scheduler_class=_name(self.scheduler) if self.scheduler is not None else None,
            criterion_state_dict=self.criterion.state_dict(),
            criterion_settings=self._criterion_settings(),
            scaler_state_dict=self.scaler.state_dict(),
            epoch=self.epoch,
            global_step=self.global_step,
            best_score=self.best_score,
            best_val_qwk=self.best_val_qwk,
            best_epoch=self.best_epoch,
            early_stop_score=self.early_stop_score,
            bad_epochs=self.bad_epochs,
            history=self.history,
            seed=self.seed,
            config=self._config(),
            class_names=self.class_names,
            dataset_fingerprints=self.dataset_info,
            rng_state=_rng_state(),
            loader_rng={key: _loader_rng(loader) for key, loader in self.loaders.items()},
        )
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    def _read_checkpoint(self, checkpoint):
        """Load a trusted local checkpoint and verify architecture and label order."""
        path = (
            self.output_dir / f"{checkpoint}.pt"
            if str(checkpoint) in ("best", "last")
            else Path(checkpoint)
        )
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        # Full local training checkpoints include Python/NumPy RNG state.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1 or state["model_signature"] != self._model_signature():
            raise ValueError("Checkpoint model architecture/configuration does not match")
        if tuple(state["class_names"]) != self.class_names:
            raise ValueError("Checkpoint class order does not match")
        return state, path

    def load_checkpoint(self, checkpoint="last", *, resume: bool = True) -> dict:
        """Load this trainer's local checkpoint; resume=True also restores run/RNG state."""
        state, _path = self._read_checkpoint(checkpoint)
        if state["criterion_settings"] != self._criterion_settings():
            raise ValueError(
                "Checkpoint loss type/settings differ; construct the matching criterion"
            )
        if resume:
            if self.optimizer is None:
                raise ValueError("Cannot resume training with a fully frozen model")
            current = self._config()
            for key in (
                "monitor",
                "mode",
                "min_delta",
                "accumulation_steps",
                "max_grad_norm",
                "amp",
                "amp_dtype",
                "metrics_kwargs",
                "scheduler_interval",
                "scheduler_monitor",
                "scheduler_spec",
                "optimizer_defaults",
                "custom_hooks",
                "trainable",
                "deterministic",
            ):
                if state["config"][key] != current[key]:
                    raise ValueError(f"Resume setting differs: {key}")
            if state["optimizer_class"] != _name(self.optimizer) or state["scheduler_class"] != (
                _name(self.scheduler) if self.scheduler is not None else None
            ):
                raise ValueError("Checkpoint optimizer/scheduler types differ")
            for split in ("train", "val"):
                old, new = state["dataset_fingerprints"][split], self.dataset_info[split]
                if old != new:
                    raise ValueError(f"Resume {split} data/loader metadata differs")
        if resume and _path.parent.resolve() != self.output_dir.resolve():
            raise ValueError(
                "Resume in the checkpoint run directory; move/copy the whole run to relocate it"
            )
        if (
            resume
            and state["best_epoch"] is not None
            and not (self.output_dir / "best.pt").is_file()
        ):
            raise ValueError(
                "Resume requires the historical best.pt alongside last.pt; restore the complete run directory"
            )
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.criterion.load_state_dict(state["criterion_state_dict"], strict=True)
        if resume:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            for key in (
                "epoch",
                "global_step",
                "best_score",
                "best_val_qwk",
                "best_epoch",
                "early_stop_score",
                "bad_epochs",
                "seed",
            ):
                setattr(self, key, state[key])
            self.history = copy.deepcopy(state["history"])
            for split, loader in self.loaders.items():
                _restore_loader_rng(loader, state["loader_rng"][split])
            _restore_rng(state["rng_state"])
            self._started = True
        return state

    def fit(
        self,
        epochs: int | None = None,
        *,
        resume_from: str | Path | None = None,
        evaluate_test: bool = True,
    ) -> list[dict]:
        """Train up to a TOTAL epoch count, optionally resuming last.pt (not extra epochs).

        Afterwards, evaluate a supplied test loader using best.pt (last.pt if no
        validation optimum was defined), saving the result in self.test_report.
        Export best/last copies to saved_weights_dir before testing, including
        when early stopping ends training. Paths are stored in self.saved_weights.
        Pass evaluate_test=False to defer testing. The return value is the epoch history.
        """
        if self.loaders["train"] is None or self.loaders["val"] is None:
            raise ValueError("fit requires training and validation loaders")
        target_epochs = self.max_epochs if epochs is None else epochs
        if (
            not isinstance(target_epochs, int)
            or isinstance(target_epochs, bool)
            or target_epochs < 1
        ):
            raise ValueError("epochs must be a positive total count")
        if resume_from is not None:
            self.load_checkpoint(resume_from, resume=True)
        if not self._started and any(
            (self.output_dir / file).exists()
            for file in ("last.pt", "best.pt", "history.json", "config.yaml")
        ):
            raise FileExistsError(
                "Run directory already contains a run; use resume_from or a new output_dir"
            )
        self.max_epochs = target_epochs
        self._prepare_output()
        self._started, self.should_stop = True, False
        self.test_report = None
        self._print_run_summary("Training run")
        if self.verbose:
            print(
                f"Epochs: {self.epoch + 1}..{target_epochs}; seed={self.seed}; "
                f"selection=val.{self.monitor}/{self.mode}; patience={self.patience}",
                flush=True,
            )
            if self.saved_weights_dir is not None:
                print(f"Saved weights directory: {self.saved_weights_dir.resolve()}", flush=True)
        self._event("fit_start", start_epoch=self.epoch + 1, total_epochs=target_epochs)
        try:
            for epoch in range(self.epoch + 1, target_epochs + 1):
                learning_rates = [group["lr"] for group in self.optimizer.param_groups]
                train, _train_meter, _indices = self._run_epoch(
                    self.loaders["train"], training=True, split="train", epoch=epoch
                )
                val, meter, indices = self._run_epoch(
                    self.loaders["val"], training=False, split="val", epoch=epoch
                )
                score = self._value(val, self.monitor)
                improved = score is not None and self._better(score, self.best_score)
                if improved:
                    self.best_score, self.best_epoch = score, epoch
                if score is not None and self._better(score, self.early_stop_score, self.min_delta):
                    self.early_stop_score, self.bad_epochs = score, 0
                else:
                    self.bad_epochs += 1
                qwk = val.get("qwk")
                if qwk is not None:
                    self.best_val_qwk = (
                        qwk if self.best_val_qwk is None else max(qwk, self.best_val_qwk)
                    )
                if self.scheduler is not None:
                    if self.scheduler_interval == "epoch":
                        self.scheduler.step()
                    elif self.scheduler_interval == "metric":
                        scheduler_value = self._value(val, self.scheduler_monitor)
                        if scheduler_value is not None:
                            self.scheduler.step(scheduler_value)
                self.epoch = epoch
                record = dict(
                    epoch=epoch,
                    train=train,
                    val=val,
                    learning_rates=learning_rates,
                    global_step=self.global_step,
                    monitored_value=score,
                    improved=improved,
                    selection_status="defined" if score is not None else "undefined",
                )
                self.history.append(record)
                if self._writer is not None:
                    for split, report in (("train", train), ("val", val)):
                        for key, value in report.items():
                            if isinstance(value, (int, float)):
                                self._writer.add_scalar(f"{split}/{key}", value, epoch)
                    self._writer.flush()
                self._event("epoch_end", **record)
                self._save_history()
                self.save_checkpoint()
                if improved:
                    self.save_checkpoint(self.output_dir / "best.pt")
                    self._export("val_best", val, meter, indices, self.loaders["val"])
                if self.verbose:
                    print(
                        f"Epoch {epoch}/{target_epochs} "
                        f"train_loss={train['loss']:.5f} val_loss={val['loss']:.5f} "
                        f"val_accuracy={self._format_metric(val.get('accuracy'), '.2%')} "
                        f"val_macro_f1={self._format_metric(val.get('macro_f1'))} "
                        f"val_{self.monitor}={self._format_metric(score, '.6f')}",
                        flush=True,
                    )
                if score is None and self.undefined_monitor == "error":
                    raise ValueError(
                        f"Validation {self.monitor} is undefined; last.pt retained, no undefined best score saved"
                    )
                if (
                    self.should_stop
                    or self.patience is not None
                    and self.bad_epochs >= self.patience
                ):
                    break
            self._event(
                "fit_end", epoch=self.epoch, best_epoch=self.best_epoch, best_score=self.best_score
            )
            self._export_saved_weights()
            if self.verbose:
                print(
                    f"Training complete: best_epoch={self.best_epoch} "
                    f"best_{self.monitor}={self._format_metric(self.best_score, '.6f')}",
                    flush=True,
                )
            if evaluate_test and self.loaders["test"] is not None:
                checkpoint = "best" if self.best_epoch is not None else "last"
                if self.verbose:
                    print(f"Evaluating test set using {checkpoint}.pt", flush=True)
                self.test_report = self.test(checkpoint=checkpoint)
                if self.verbose:
                    print(
                        f"Test loss={self.test_report['loss']:.5f} "
                        f"accuracy={self._format_metric(self.test_report.get('accuracy'), '.2%')} "
                        f"macro_f1={self._format_metric(self.test_report.get('macro_f1'))}",
                        flush=True,
                    )
            return self.history
        finally:
            if self._writer is not None:
                self._writer.close()
                self._writer = None

    def _export(self, split, report, meter, indices, loader):
        """Write split metrics and ordered per-example probability predictions."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", split):
            raise ValueError("split must be a simple name containing letters, numbers, '_' or '-'")
        directory = self.output_dir / "evaluations"
        _write_json(directory / f"{split}_metrics.json", report)
        if not self.save_predictions:
            return
        data = meter.get_predictions()
        if data["probabilities"] is None:
            raise ValueError("Prediction export requires class probability matrices")
        base = _base_dataset(loader.dataset)
        root = getattr(base, "root", None)
        with (directory / f"{split}_predictions.csv").open("w", newline="") as file:
            fields = [
                "sample_index",
                "sample_id",
                "image_path",
                "true_label",
                "predicted_label",
            ] + [f"p{i}" for i in range(self.num_classes)]
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row, index in enumerate(indices):
                sample = _sample(loader.dataset, index)
                path = sample[0] if sample is not None else ""
                if path and root is not None:
                    path = os.path.relpath(path, root)
                record = dict(
                    sample_index=index,
                    sample_id=path or str(index),
                    image_path=path,
                    true_label=int(data["targets"][row]),
                    predicted_label=int(data["predictions"][row]),
                )
                record.update(
                    {f"p{i}": float(data["probabilities"][row, i]) for i in range(self.num_classes)}
                )
                writer.writerow(record)

    def evaluate(
        self,
        loader: DataLoader | None = None,
        *,
        split: str = "test",
        checkpoint: str | Path | None = "best",
        save: bool = True,
        compute_loss: bool = True,
    ) -> dict:
        """Evaluate a frozen checkpoint on test/external data without optimizer updates.

        None evaluates current weights. Requested checkpoint weights/loss buffers
        are temporary; current model, loss, modes and RNGs are restored afterwards.
        Pass compute_loss=False for metrics-only evaluation with a different loss
        configuration. Sequential, deterministic loaders retain all examples.
        """
        loader = loader if loader is not None else self.loaders["test"]
        _check_eval_loader(loader)
        classes = getattr(_base_dataset(loader.dataset), "classes", None)
        if classes is not None and tuple(classes) != self.class_names:
            raise ValueError("Evaluation label order differs from training")
        base = _base_dataset(loader.dataset)
        train_info = self.dataset_info.get("train")

        def check_normalization(train_info):
            if train_info is None:
                return
            for field in ("normalization_mean", "normalization_std"):
                if train_info[field] != _jsonable(getattr(base, field, None)):
                    raise ValueError(
                        "Evaluation normalization must match the saved training normalization"
                    )

        check_normalization(train_info)
        rng, loader_rng = _rng_state(), _loader_rng(loader)
        model_state = (
            {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            if checkpoint is not None
            else None
        )
        loss_state = (
            {
                key: value.detach().cpu().clone()
                for key, value in self.criterion.state_dict().items()
            }
            if checkpoint is not None and compute_loss
            else None
        )
        try:
            checkpoint_path = None
            if checkpoint is not None:
                state, checkpoint_path = self._read_checkpoint(checkpoint)
                check_normalization(state["dataset_fingerprints"].get("train"))
                if compute_loss and state["criterion_settings"] != self._criterion_settings():
                    raise ValueError(
                        "Evaluation loss differs from checkpoint; supply the matching criterion or compute_loss=False"
                    )
                self.model.load_state_dict(state["model_state_dict"], strict=True)
                if compute_loss:
                    self.criterion.load_state_dict(state["criterion_state_dict"], strict=True)
            self._print_run_summary(
                f"Evaluation / inference: {split}",
                loaders={split: loader},
                checkpoint=checkpoint_path if checkpoint_path is not None else "current weights",
            )
            report, meter, indices = self._run_epoch(
                loader, training=False, split=split, epoch=self.epoch, compute_loss=compute_loss
            )
            report["checkpoint"] = str(checkpoint_path) if checkpoint_path is not None else None
            report["split"] = split
            if save:
                self._export(split, report, meter, indices, loader)
            self._event("evaluation_end", split=split, report=report)
            return report
        finally:
            if model_state is not None:
                self.model.load_state_dict(model_state, strict=True)
            if loss_state is not None:
                self.criterion.load_state_dict(loss_state, strict=True)
            _restore_loader_rng(loader, loader_rng)
            _restore_rng(rng)

    def test(
        self,
        *,
        checkpoint: str | Path | None = "best",
        save: bool = True,
        compute_loss: bool = True,
    ) -> dict:
        """Evaluate the configured test loader (best validation checkpoint by default)."""
        return self.evaluate(
            split="test", checkpoint=checkpoint, save=save, compute_loss=compute_loss
        )
