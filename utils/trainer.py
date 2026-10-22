"""Scratch training with validation selection, resume and mandatory TensorBoard logs."""

import platform
from copy import deepcopy
from pathlib import Path

import medmnist
import numpy as np
import timm
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from FDMamba import build_model, model_name

from .config import resolve_config, run_name
from .data import class_names, dataset_info, make_loader
from .engine import run_epoch
from .evaluation import save_predictions
from .runtime import (
    make_scheduler,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    select_device,
    write_csv,
    write_json,
)


class Trainer:
    """Train one model/dataset, log its experiment and export selected inference weights."""

    def __init__(self, config, dataset, *, name=None, download=False, verbose=True):
        self.config = resolve_config(config)
        self.dataset = dataset
        self.classes = class_names(dataset)
        self.settings = self.config["training"]
        self.device = select_device(self.settings["device"])
        model_config, data = self.config["model"], self.config["data"]
        if model_config["variant"] in ("B2", "B3") and self.device.type != "cuda":
            raise RuntimeError("B2/B3 need a CUDA device and real Mamba extensions")
        canonical = model_name(model_config["backbone"], model_config["variant"])
        self.name = run_name(name or canonical)
        self.output_dir = Path(
            self.settings["output_dir"]
            or (
                f"experiments/{dataset}/{self.name}/size{data['size']}_input{data['image_size']}_seed{self.settings['seed']}"
            )
        ).expanduser()
        custom = f"_{self.name}" if self.name != canonical else ""
        self.weights_path = Path(self.settings["weights_dir"]).expanduser() / (
            f"{canonical}_{dataset}{custom}_size{data['size']}_input{data['image_size']}"
            f"_seed{self.settings['seed']}.pt"
        )
        self.verbose = verbose
        seed_everything(self.settings["seed"])
        self.model = build_model(**model_config, num_classes=len(self.classes)).to(self.device)
        self.loaders = {
            split: make_loader(
                dataset,
                split,
                data,
                seed=self.settings["seed"] + index,
                download=download and split == "train",
            )
            for index, split in enumerate(("train", "val"))
        }
        counts = np.bincount(
            self.loaders["train"].dataset.labels[:, 0], minlength=len(self.classes)
        )
        weights = None
        if self.settings["class_weighted"]:
            if (counts == 0).any():
                raise ValueError(
                    "Weighted CE requires every class in training; set class_weighted: false otherwise"
                )
            weights = torch.tensor(
                counts.sum() / (len(counts) * counts), dtype=torch.float32, device=self.device
            )
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=self.settings["label_smoothing"],
            reduction="sum",
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.settings["lr"],
            weight_decay=self.settings["weight_decay"],
        )
        self.scheduler = make_scheduler(
            self.optimizer, self.settings["epochs"], self.settings["warmup_epochs"]
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.settings["amp"] and self.device.type == "cuda"
        )
        self.epoch, self.best_epoch, self.bad_epochs = 0, None, 0
        self.best_score, self.history, self.test_report = None, [], None
        info = dataset_info(dataset)
        self.data_metadata = dict(
            version=medmnist.__version__,
            task=info["task"],
            labels=info["label"],
            n_samples=info["n_samples"],
            size=data["size"],
            license=info["license"],
        )

    def _save(self, filename):
        """Save weights, preprocessing and the complete epoch-boundary training state."""
        state = dict(
            format_version=2,
            name=self.name,
            dataset=self.dataset,
            class_names=self.classes,
            config=self.config,
            model_config=self.model.build_config,
            data_metadata=self.data_metadata,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            scaler_state=self.scaler.state_dict(),
            epoch=self.epoch,
            best_epoch=self.best_epoch,
            best_score=self.best_score,
            bad_epochs=self.bad_epochs,
            history=self.history,
            rng=rng_state(self.loaders),
        )
        save_checkpoint(self.output_dir / filename, state)

    def _export_weights(self, state, selected):
        """Export the selected CPU weights and reconstruction metadata without optimizer state."""
        exported = {
            key: state[key]
            for key in (
                "format_version",
                "name",
                "dataset",
                "class_names",
                "model_config",
                "data_metadata",
                "model_state",
                "epoch",
            )
        }
        exported.update(
            config=self.config,
            selection=dict(
                monitor=self.settings["monitor"],
                score=self.best_score,
                source_checkpoint=str(selected.resolve()),
            ),
        )
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self.weights_path, exported)
        if self.verbose:
            print(f"Saved weights (epoch {state['epoch']}): {self.weights_path}")

    def _resume(self):
        """Restore last.pt only when model, data and training settings still match."""
        state = torch.load(self.output_dir / "last.pt", map_location="cpu", weights_only=False)
        if state.get("format_version") != 2:
            raise ValueError("Checkpoint format differs; use a current FDMamba training checkpoint")
        previous, current = deepcopy(state["config"]), deepcopy(self.config)
        # Artifact locations may change without changing the training run.
        for key in ("output_dir", "weights_dir"):
            previous["training"].pop(key, None)
            current["training"].pop(key, None)
        if (
            previous != current
            or state["dataset"] != self.dataset
            or state["name"] != self.name
            or state["class_names"] != self.classes
            or state["data_metadata"] != self.data_metadata
        ):
            raise ValueError(
                "Resume configuration/dataset differs; use the original settings and total epoch budget"
            )
        if state["best_epoch"] is not None and not (self.output_dir / "best.pt").exists():
            raise FileNotFoundError(
                "Resume needs best.pt alongside last.pt; keep the complete experiment folder"
            )
        self.model.load_state_dict(state["model_state"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.scaler.load_state_dict(state["scaler_state"])
        for field in ("epoch", "best_epoch", "best_score", "bad_epochs", "history"):
            setattr(self, field, state[field])
        restore_rng(state["rng"], self.loaders)

    def _write_history(self):
        """Store full reports in JSON and scalar epoch metrics in CSV."""
        write_json(self.output_dir / "history.json", self.history)
        rows = []
        for item in self.history:
            row = dict(epoch=item["epoch"], lr=item["lr"])
            for split in ("train", "val"):
                row.update(
                    {
                        f"{split}_{key}": value
                        for key, value in item[split].items()
                        if value is None or isinstance(value, (int, float))
                    }
                )
            rows.append(row)
        write_csv(self.output_dir / "history.csv", rows)

    @staticmethod
    def _log(writer, split, report, epoch):
        """Send every defined scalar measurement to TensorBoard."""
        for key, value in report.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"{split}/{key}", value, epoch)

    def fit(self, *, resume=False, stop_after=None):
        """Train to the configured budget; stop_after permits a resumable smoke run."""
        if resume:
            self._resume()
        elif any(
            (self.output_dir / name).exists() for name in ("config.yaml", "last.pt", "best.pt")
        ):
            raise FileExistsError(f"Experiment exists: {self.output_dir}; use --resume or --name")
        elif self.weights_path.exists():
            raise FileExistsError(
                f"Saved weights exist: {self.weights_path}; use --name for a new run "
                "or --resume with the original experiment"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        metadata = dict(
            name=self.name,
            dataset=self.dataset,
            classes=self.classes,
            data=self.data_metadata,
            device=str(self.device),
            gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            python=platform.python_version(),
            torch=torch.__version__,
            timm=timm.__version__,
            parameters=sum(p.numel() for p in self.model.parameters()),
            class_counts={
                split: np.bincount(
                    loader.dataset.labels[:, 0], minlength=len(self.classes)
                ).tolist()
                for split, loader in self.loaders.items()
            },
        )
        write_json(self.output_dir / "metadata.json", metadata)
        if self.verbose:
            print(f"{self.name} | {self.dataset} | {len(self.classes)} classes | {self.device}")
            print(
                f"Parameters: {metadata['parameters']:,}; train/val counts: {metadata['class_counts']}"
            )
            print(f"Experiment: {self.output_dir}\nTensorBoard: {self.output_dir / 'tensorboard'}")
            print(f"Final weights: {self.weights_path}")
        monitor = self.settings["monitor"]
        limit = (
            self.settings["epochs"]
            if stop_after is None
            else min(stop_after, self.settings["epochs"])
        )
        patience = self.settings["patience"]
        # Always open/close the writer, including resumed and interrupted training.
        with SummaryWriter(
            str(self.output_dir / "tensorboard"), purge_step=self.epoch + 1 if resume else None
        ) as writer:
            writer.add_text("run/config", yaml.safe_dump(self.config), 0)
            writer.add_text("run/metadata", str(metadata), 0)
            for epoch in range(self.epoch + 1, limit + 1):
                if patience is not None and self.bad_epochs >= patience:
                    break
                lr = self.optimizer.param_groups[0]["lr"]
                train_report, _, _ = run_epoch(
                    self.model,
                    self.loaders["train"],
                    self.device,
                    self.classes,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    amp=self.settings["amp"],
                    accumulation_steps=self.settings["accumulation_steps"],
                    max_grad_norm=self.settings["max_grad_norm"],
                    description=f"Train {epoch}",
                    verbose=self.verbose,
                )
                val_report, _, _ = run_epoch(
                    self.model,
                    self.loaders["val"],
                    self.device,
                    self.classes,
                    criterion=self.criterion,
                    amp=self.settings["amp"],
                    description=f"Val {epoch}",
                    verbose=self.verbose,
                )
                score = val_report[monitor]
                improved = score is not None and (
                    self.best_score is None
                    or (score < self.best_score if monitor == "loss" else score > self.best_score)
                )
                self.epoch = epoch
                if improved:
                    self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
                else:
                    self.bad_epochs += 1
                self.history.append(dict(epoch=epoch, lr=lr, train=train_report, val=val_report))
                self.scheduler.step()
                self._write_history()
                self._save("last.pt")
                if improved:
                    self._save("best.pt")
                self._log(writer, "train", train_report, epoch)
                self._log(writer, "val", val_report, epoch)
                writer.add_scalar("train/learning_rate", lr, epoch)
                writer.flush()
                if self.verbose:
                    print(
                        f"Epoch {epoch}/{self.settings['epochs']} | loss={train_report['loss']:.4f} "
                        f"val_loss={val_report['loss']:.4f} accuracy={val_report['accuracy']:.4f} "
                        f"macro_f1={val_report['macro_f1']:.4f} balanced_accuracy={val_report['balanced_accuracy']:.4f}"
                    )
            finished = self.epoch >= self.settings["epochs"] or (
                patience is not None and self.bad_epochs >= patience
            )
            if finished:
                selected = self.output_dir / (
                    "best.pt" if self.best_epoch is not None else "last.pt"
                )
                state = torch.load(selected, map_location="cpu", weights_only=False)
                self.model.load_state_dict(state["model_state"], strict=True)
                self._export_weights(state, selected)
                del state
                test_loader = make_loader(self.dataset, "test", self.config["data"])
                report, targets, probabilities = run_epoch(
                    self.model,
                    test_loader,
                    self.device,
                    self.classes,
                    criterion=self.criterion,
                    description="Test",
                    verbose=self.verbose,
                )
                report.update(
                    dataset=self.dataset,
                    name=self.name,
                    checkpoint=str(selected),
                    saved_weight=str(self.weights_path),
                )
                write_json(self.output_dir / "test_metrics.json", report)
                save_predictions(
                    self.output_dir / "test_predictions.csv",
                    targets,
                    probabilities,
                    dataset=self.dataset,
                    split="test",
                    size=self.config["data"]["size"],
                )
                self._log(writer, "test", report, self.epoch)
                self.test_report = report
                if self.verbose:
                    print(
                        f"Test accuracy={report['accuracy']:.4f}, macro_f1={report['macro_f1']:.4f}, AUC={report['auc']}"
                    )
        return self.history
