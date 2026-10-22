"""Offline checks of data, architecture, TensorBoard, inference and benchmarking."""

import csv
import hashlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import medmnist
import numpy as np
import torch
from medmnist.evaluator import getAUC
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from benchmark import benchmark
from download_data import download_datasets
from FDMamba import build_model
from utils.config import load_config, resolve_config
from utils.data import DEFAULT_DATASETS, class_names, dataset_info, load_dataset, make_loader
from utils.engine import run_epoch
from utils.evaluation import evaluate_checkpoint, predict_image
from utils.metrics import classification_metrics
from utils.trainer import Trainer


def setUpModule():
    torch.set_num_threads(2)


class DatasetTests(unittest.TestCase):
    """Tiny official-format archives keep workflow tests independent of downloads."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in (*DEFAULT_DATASETS, "pneumoniamnist"):
            info = deepcopy(medmnist.INFO[name])
            count = len(info["label"])
            arrays = {}
            for split, repeats in (("train", 2), ("val", 1), ("test", 1)):
                labels = np.tile(np.arange(count), repeats).reshape(-1, 1)
                info["n_samples"][split] = len(labels)
                shape = (len(labels), 28, 28) + ((3,) if info["n_channels"] == 3 else ())
                arrays[f"{split}_images"] = np.random.default_rng(len(labels)).integers(
                    0, 256, shape, dtype=np.uint8
                )
                arrays[f"{split}_labels"] = labels
            path = self.root / f"{name}.npz"
            np.savez_compressed(path, **arrays)
            info["MD5"] = hashlib.md5(path.read_bytes()).hexdigest()
            mock = patch.dict(medmnist.INFO, {name: info})
            mock.start()
            self.addCleanup(mock.stop)

    def config(self, output="run"):
        return resolve_config(
            {
                "model": {"backbone": "mobilenetv3", "variant": "B0"},
                "data": {
                    "root": str(self.root),
                    "image_size": 32,
                    "batch_size": 2,
                    "mean": [0.5] * 3,
                    "std": [0.5] * 3,
                },
                "training": {
                    "epochs": 2,
                    "warmup_epochs": 0,
                    "device": "cpu",
                    "output_dir": str(self.root / output),
                    "weights_dir": str(self.root / "saved_weight"),
                },
            }
        )

    def test_official_datasets_are_used_directly(self):
        for name, count in zip(DEFAULT_DATASETS, (9, 7, 8, 11)):
            with self.subTest(name=name):
                data = self.config()["data"]
                loader = make_loader(name, "test", data)
                self.assertIsInstance(
                    loader.dataset, getattr(medmnist, dataset_info(name)["python_class"])
                )
                self.assertEqual(len(class_names(name)), count)
                images, labels = next(iter(loader))
                self.assertEqual(images.shape, (2, 3, 32, 32))
                self.assertEqual(labels.shape, (2,))
                self.assertEqual(labels.dtype, torch.long)
                self.assertIsInstance(loader.sampler, SequentialSampler)
                self.assertFalse(loader.drop_last)

    def test_grayscale_and_eval_preprocessing_are_deterministic(self):
        dataset = load_dataset("organamnist", "test", root=self.root, augment=True)
        image, target = dataset[0]
        self.assertIsInstance(target, int)
        torch.testing.assert_close(image[0], image[1])
        torch.testing.assert_close(image, dataset[0][0])
        self.assertNotIn("Random", repr(dataset.transform))

    def test_download_uses_medmnist_cache_without_network(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected download")):
            download_datasets(["bloodmnist"], root=self.root)
        self.assertTrue((self.root / "bloodmnist.npz").exists())

    def test_download_resumes_partial_archive(self):
        source = self.root / "bloodmnist.npz"
        payload = source.read_bytes()
        source.unlink()
        cutoff = len(payload) // 3
        source.write_bytes(payload[:cutoff])  # MedMNIST leaves partial data here.

        class PartialResponse(io.BytesIO):
            status = 206

            def __init__(self, content):
                super().__init__(content[cutoff:])
                self.headers = {
                    "Content-Length": str(len(content) - cutoff),
                    "Content-Range": f"bytes {cutoff}-{len(content) - 1}/{len(content)}",
                }

            def getcode(self):
                return self.status

        def resume(request):
            self.assertEqual(request.get_header("Range"), f"bytes={cutoff}-")
            return PartialResponse(payload)

        with patch("urllib.request.urlopen", side_effect=resume) as urlopen:
            download_datasets(["bloodmnist"], root=self.root)

        self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(source.read_bytes(), payload)
        self.assertFalse((self.root / "bloodmnist.npz.part").exists())

    def test_incompatible_tasks_and_missing_data_fail(self):
        for name in ("chestmnist", "organmnist3d", "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dataset_info(name)
        with self.assertRaises(RuntimeError):
            load_dataset("pathmnist", "train", root=self.root / "empty")
        with self.assertRaises(ValueError):
            load_dataset("pathmnist", "train", root=self.root, size=32)

    def test_resume_tensorboard_inference_and_custom_benchmark(self):
        config = self.config()
        with patch("utils.trainer.make_loader", wraps=make_loader) as loaders:
            first = Trainer(config, "pneumoniamnist", name="MobileNet-v1", verbose=False)
            first.fit(stop_after=1)
            self.assertIsNone(first.test_report)
            self.assertFalse(first.weights_path.exists())
            self.assertEqual([call.args[1] for call in loaders.call_args_list], ["train", "val"])
        resumed = Trainer(config, "pneumoniamnist", name="MobileNet-v1", verbose=False)
        resumed.fit(resume=True)
        self.assertEqual(resumed.epoch, 2)
        self.assertEqual(resumed.test_report["num_samples"], 2)
        log = EventAccumulator(str(resumed.output_dir / "tensorboard")).Reload()
        tags = log.Tags()["scalars"]
        for tag in (
            "train/loss",
            "val/macro_f1",
            "val/auc",
            "test/accuracy",
            "train/learning_rate",
        ):
            self.assertIn(tag, tags)
        self.assertEqual([event.step for event in log.Scalars("train/loss")], [1, 2])
        # Exact epoch-boundary resume with zero workers and unchanged total budget.
        continuous = Trainer(self.config("continuous"), "pneumoniamnist", verbose=False)
        continuous.fit()
        actual = torch.load(resumed.output_dir / "last.pt", weights_only=False)["model_state"]
        expected = torch.load(continuous.output_dir / "last.pt", weights_only=False)["model_state"]
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        checkpoint = resumed.weights_path
        self.assertEqual(
            checkpoint.name,
            "FDMamba-MobileNetV3-B0_pneumoniamnist_MobileNet-v1_size28_input32_seed42.pt",
        )
        self.assertEqual(
            continuous.weights_path.name,
            "FDMamba-MobileNetV3-B0_pneumoniamnist_size28_input32_seed42.pt",
        )
        exported = torch.load(checkpoint, map_location="cpu", weights_only=False)
        selected = torch.load(resumed.output_dir / "best.pt", weights_only=False)
        self.assertEqual(exported["epoch"], resumed.best_epoch)
        self.assertEqual(exported["selection"]["score"], resumed.best_score)
        for key in ("optimizer_state", "scheduler_state", "scaler_state", "rng", "history"):
            self.assertNotIn(key, exported)
        for key in exported["model_state"]:
            torch.testing.assert_close(
                exported["model_state"][key], selected["model_state"][key], atol=0, rtol=0
            )
        with patch("utils.evaluation.make_loader", wraps=make_loader) as loaders:
            report = evaluate_checkpoint(
                checkpoint,
                "pneumoniamnist",
                output_dir=self.root / "evaluation",
                device="cpu",
                verbose=False,
            )
            self.assertEqual(loaders.call_args.args[1], "test")
        self.assertEqual(report["accuracy"], resumed.test_report["accuracy"])
        image_path = self.root / "image.png"
        source = load_dataset("pneumoniamnist", "test", root=self.root)
        Image.fromarray(source.imgs[0]).save(image_path)
        inference = predict_image(
            checkpoint, image_path, model="MobileNet-v1", device="cpu", top_k=2
        )
        with (self.root / "evaluation/predictions.csv").open() as handle:
            row = next(csv.DictReader(handle))
        for item in inference["predictions"]:
            self.assertAlmostEqual(item["probability"], float(row[f"p{item['label']}"]), places=6)
        entries = [
            dict(name=name, checkpoint=str(checkpoint), dataset="pneumoniamnist")
            for name in ("custom-a", "custom-b")
        ]
        result = benchmark(entries, output_dir=self.root / "benchmark", device="cpu", verbose=False)
        self.assertEqual([row["name"] for row in result], ["custom-a", "custom-b"])
        self.assertTrue((self.root / "benchmark/summary.csv").exists())
        with self.assertRaises(ValueError):
            predict_image(checkpoint, image_path, model="resnet50", device="cpu")
        with self.assertRaises(ValueError):
            evaluate_checkpoint(checkpoint, "bloodmnist", device="cpu", verbose=False)
        with self.assertRaises(ValueError):
            benchmark([entries[0], entries[0]], output_dir=self.root / "benchmark", verbose=False)
        with self.assertRaises(FileExistsError):
            Trainer(config, "pneumoniamnist", name="MobileNet-v1", verbose=False).fit()
        with self.assertRaisesRegex(FileExistsError, "Saved weights exist"):
            Trainer(self.config("new-folder"), "pneumoniamnist", verbose=False).fit()
        modified = deepcopy(config)
        modified["training"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "Resume configuration"):
            Trainer(modified, "pneumoniamnist", name="MobileNet-v1", verbose=False).fit(resume=True)

    def test_early_stop_exports_last_when_validation_monitor_is_undefined(self):
        path = self.root / "pneumoniamnist.npz"
        with np.load(path) as archive:
            arrays = dict(archive)
        # With only one validation class, AUC cannot select a best checkpoint.
        arrays["val_labels"][:] = 0
        np.savez_compressed(path, **arrays)
        config = self.config()
        config["training"].update(epochs=3, patience=1, monitor="auc")
        trainer = Trainer(config, "pneumoniamnist", verbose=False)
        trainer.fit()
        self.assertEqual(trainer.epoch, 1)
        self.assertIsNone(trainer.best_epoch)
        self.assertFalse((trainer.output_dir / "best.pt").exists())
        exported = torch.load(trainer.weights_path, map_location="cpu", weights_only=False)
        self.assertEqual(exported["epoch"], 1)
        self.assertEqual(Path(exported["selection"]["source_checkpoint"]).name, "last.pt")
        self.assertIsNone(exported["selection"]["score"])
        last = torch.load(trainer.output_dir / "last.pt", weights_only=False)
        for key in exported["model_state"]:
            torch.testing.assert_close(
                exported["model_state"][key], last["model_state"][key], atol=0, rtol=0
            )
        # A completed/early-stopped experiment can regenerate its export without retraining.
        config["training"]["weights_dir"] = str(self.root / "relocated_weights")
        resumed = Trainer(config, "pneumoniamnist", verbose=False)
        resumed.fit(resume=True)
        self.assertEqual(resumed.epoch, 1)
        self.assertTrue(resumed.weights_path.exists())


class MetricAndEngineTests(unittest.TestCase):
    """Metrics and accumulated weighted CE follow their standard definitions."""

    def test_auc_matches_official_evaluator(self):
        for classes in (2, 3):
            labels = np.tile(np.arange(classes), 2)
            scores = np.random.default_rng(42).random((len(labels), classes))
            scores /= scores.sum(axis=1, keepdims=True)
            report = classification_metrics(labels, scores, [str(i) for i in range(classes)])
            task = "binary-class" if classes == 2 else "multi-class"
            self.assertAlmostEqual(report["auc"], getAUC(labels, scores, task))
        absent = classification_metrics([0, 0], [[0.7, 0.3], [0.2, 0.8]], ["a", "b"])
        self.assertIsNone(absent["auc"])
        self.assertIn("auc", absent["undefined_metrics"])

    def test_weighted_accumulation_matches_full_windows(self):
        torch.manual_seed(42)
        model = nn.Linear(4, 3)
        reference = deepcopy(model)
        images, labels = torch.randn(5, 4), torch.tensor([0, 1, 2, 1, 0])
        weights = torch.tensor([1.0, 2.0, 4.0])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        manual = torch.optim.SGD(reference.parameters(), lr=0.01)
        run_epoch(
            model,
            DataLoader(TensorDataset(images, labels), batch_size=2),
            torch.device("cpu"),
            ["a", "b", "c"],
            criterion=nn.CrossEntropyLoss(weight=weights, reduction="sum"),
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            accumulation_steps=2,
            verbose=False,
        )
        for start, end in ((0, 4), (4, 5)):
            manual.zero_grad()
            nn.functional.cross_entropy(
                reference(images[start:end]), labels[start:end], weight=weights
            ).backward()
            manual.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)


class ModelAndConfigTests(unittest.TestCase):
    """All supported architecture variants retain their native model contracts."""

    def test_presets_and_unknown_settings(self):
        for path in ("configs/train.yaml", "configs/efficientnet.yaml", "configs/resnet.yaml"):
            config = load_config(path)
            self.assertEqual(config["model"]["variant"], "B3")
        for config in (
            {"unknown": {}},
            {"training": {"epohs": 1}},
            {"training": {"epochs": 1}},
            {"data": {"mean": [0.5] * 3}},
            {"model": {"pretrained": True}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                resolve_config(config)

    def test_all_baseline_and_fdconv_models_forward_backward(self):
        sites = {
            "mobilenetv3": "blocks.2.2.conv_pwl",
            "efficientnet_b0": "blocks.2.0.conv_pwl",
            "resnet50": "layer3.3.conv2",
        }
        for backbone, site in sites.items():
            for variant in ("B0", "B1"):
                with self.subTest(backbone=backbone, variant=variant):
                    model = build_model(backbone=backbone, variant=variant, num_classes=7)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    # ResNet's zero-initialized residual scale learns on the first
                    # step; upstream convolution gradients become active afterwards.
                    for _ in range(2):
                        optimizer.zero_grad()
                        logits = model(torch.randn(2, 3, 64, 64))
                        self.assertEqual(logits.shape, (2, 7))
                        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                        optimizer.step()
                    self.assertTrue(
                        all(
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )
                    if variant == "B1":
                        self.assertEqual(model.replacement.path, site)
                        spectral = [
                            p for name, p in model.named_parameters() if "dft_weight" in name
                        ]
                        self.assertTrue(spectral)
                        self.assertTrue(
                            all(p.grad is not None and p.grad.abs().sum() > 0 for p in spectral)
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "Real Mamba requires CUDA")
    def test_real_mamba_models(self):
        for backbone in ("mobilenetv3", "efficientnet_b0", "resnet50"):
            for variant in ("B2", "B3"):
                with self.subTest(backbone=backbone, variant=variant):
                    model = build_model(backbone=backbone, variant=variant, num_classes=9).cuda()
                    logits = model(torch.randn(2, 3, 224, 224, device="cuda"))
                    logits.square().mean().backward()
                    self.assertEqual(logits.shape, (2, 9))
                    self.assertTrue(
                        all(
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )


if __name__ == "__main__":
    unittest.main()
