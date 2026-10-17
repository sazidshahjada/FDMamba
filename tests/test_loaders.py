"""Synthetic ImageFolder checks; no real dataset or CUDA is required."""
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from utils.loaders import create_dataloaders, compute_train_mean_std, visualize_images


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(plt.close, 'all')
        self.root = Path(self.temporary.name)
        self.paths = {split: self.root / split for split in ('train', 'val', 'test')}
        labels = {'train': range(5), 'val': (0, 4), 'test': (1, 3, 4)}
        for split, classes in labels.items():
            for label in classes:
                directory = self.paths[split] / str(label)
                directory.mkdir(parents=True)
                for index in range(2):
                    # Nonconstant color across files, with one grayscale source.
                    image = Image.new('RGB', (17, 19), (20 + 30 * label, 40 + 30 * index, 80 + 20 * label))
                    if split == 'train' and label == 0 and index == 0:
                        image = image.convert('L')
                    image.save(directory / f'{index}.png')

    def loaders(self, **kwargs):
        options = dict(image_size=(16, 20), batch_size=3, augment=False)
        options.update(kwargs)
        return create_dataloaders(**{f'{split}_path': path for split, path in self.paths.items()}, **options)

    def test_rgb_shapes_fixed_grade_mapping_and_samplers(self):
        loaders = self.loaders()
        self.assertIsInstance(loaders['train'].sampler, RandomSampler)
        for split, loader in loaders.items():
            self.assertEqual(loader.dataset.class_to_idx, {str(i): i for i in range(5)})
            images, labels = next(iter(loader))
            self.assertEqual(images.shape, (3, 3, 16, 20))
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(labels.dtype, torch.long)
            self.assertTrue(((images >= 0) & (images <= 1)).all())
            self.assertEqual(sum(loader.dataset.class_counts.values()), len(loader.dataset))
            if split != 'train':
                self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(loaders['val'].dataset.targets, [0, 0, 4, 4])
        self.assertEqual(loaders['val'].dataset.class_counts['2'], 0)

    def test_eval_is_deterministic_and_keeps_final_batch(self):
        loaders = self.loaders(augment=True, drop_last_train=True)
        self.assertTrue(any(isinstance(t, transforms.RandomRotation)
                            for t in loaders['train'].dataset.transform.transforms))
        self.assertTrue(loaders['train'].drop_last)
        for split in ('val', 'test'):
            loader = loaders[split]
            first = list(loader)
            second = list(loader)
            for (a, labels_a), (b, labels_b) in zip(first, second):
                torch.testing.assert_close(a, b)
                torch.testing.assert_close(labels_a, labels_b)
            self.assertEqual(sum(len(labels) for _, labels in first), len(loader.dataset))
            self.assertFalse(loader.drop_last)

    def test_shuffle_seed_and_worker_loading(self):
        first, second = self.loaders(seed=123), self.loaders(seed=123)
        self.assertEqual(list(first['train'].sampler), list(second['train'].sampler))
        loader = self.loaders(num_workers=1)['val']
        images, labels = next(iter(loader))
        self.assertEqual(images.shape, (3, 3, 16, 20))
        self.assertEqual(labels.tolist(), [0, 0, 4])

    def test_train_statistics_match_pixels_and_normalization(self):
        mean, std = compute_train_mean_std(self.paths['train'], image_size=(16, 20), batch_size=3)
        dataset = self.loaders()['train'].dataset
        pixels = torch.stack([dataset[i][0] for i in range(len(dataset))]).double()
        torch.testing.assert_close(torch.tensor(mean, dtype=torch.float64), pixels.mean(dim=(0, 2, 3)))
        torch.testing.assert_close(torch.tensor(std, dtype=torch.float64), pixels.std(dim=(0, 2, 3), correction=0))
        normalized = self.loaders(mean=mean, std=std)['val'].dataset
        raw = self.loaders()['val'].dataset[0][0]
        expected = (raw - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
        torch.testing.assert_close(normalized[0][0], expected)

    def test_grid_denormalizes_saves_and_preserves_rng(self):
        loaders = self.loaders(augment=True, mean=(0.3, 0.4, 0.5), std=(0.2, 0.3, 0.4))
        output = self.root / 'figures' / 'grid.png'
        torch_state = torch.random.get_rng_state().clone()
        shuffle_state = loaders['train'].generator.get_state().clone()
        figure, axes = visualize_images(loaders, rows=2, cols=3, split='val', show=False, save_path=output)
        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(sum(len(axis.images) for axis in axes.flat), 4)
        self.assertTrue(output.is_file())
        first = np.asarray(axes[0, 0].images[0].get_array())
        source_colors = {(20, 40, 80), (20, 70, 80), (140, 40, 160), (140, 70, 160)}
        self.assertIn(tuple(np.rint(first[0, 0] * 255).astype(int)), source_colors)
        visualize_images(loaders, rows=1, cols=1, split='train', show=False)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertTrue(torch.equal(loaders['train'].generator.get_state(), shuffle_state))
        plt.close(figure)

    def test_custom_folder_order_and_input_errors(self):
        loaders = self.loaders(class_names=('4', '3', '2', '1', '0'))
        self.assertEqual(loaders['val'].dataset.class_to_idx['4'], 0)
        for kwargs in ({'image_size': 0}, {'batch_size': 0}, {'num_workers': -1},
                       {'batch_size': 100, 'drop_last_train': True},
                       {'mean': (0.5,) * 3}, {'mean': (0.5,) * 3, 'std': (0.0,) * 3},
                       {'class_names': ('0', '1')}, {'class_names': ('0', '0')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.loaders(**kwargs)
        with self.assertRaises(FileNotFoundError):
            create_dataloaders(self.root / 'absent', self.paths['val'], self.paths['test'])
        with self.assertRaises(ValueError):
            create_dataloaders(self.paths['train'], self.paths['train'], self.paths['test'])
        with self.assertRaises(ValueError):
            visualize_images(loaders, split='unknown', show=False)
        with self.assertRaises(ValueError):
            visualize_images(loaders, rows=0, show=False)

    def test_empty_split_fails_clearly(self):
        empty = self.root / 'empty'
        (empty / '0').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'No supported images'):
            create_dataloaders(self.paths['train'], self.paths['val'], empty)


if __name__ == '__main__':
    unittest.main()
