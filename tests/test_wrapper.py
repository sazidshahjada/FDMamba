"""Classifier replacement and gradient checks for the shared backbone wrapper."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from fdmambadr.backbone import Backbone
from fdmambadr.fdconv_models import FDConvBackbone
from fdmambadr.hybrid_models import HybridBackbone
from fdmambadr.mamba_models import MambaBackbone
from fdmambadr.wrapper import BackboneWrapper


torch.set_num_threads(2)


class WrapperTests(unittest.TestCase):
    def test_baseline_and_fdconv_classes_keep_all_retained_parameters_in_use(self):
        # Includes MobileNet's 960 spatial channels -> 1280 native embedding.
        for backbone_class in (Backbone, FDConvBackbone):
            for name in Backbone.supported_names():
                with self.subTest(backbone=backbone_class.__name__, name=name):
                    model = BackboneWrapper(
                        backbone_class, name=name, hidden_dims=(32, 16),
                        activation=nn.GELU, normalization="layernorm",
                        dropout=(0.2, 0.0),
                    ).train()
                    native = next(m for m in model.backbone.modules()
                                  if callable(getattr(m, 'get_classifier', None)))
                    self.assertIsInstance(native.get_classifier(), nn.Identity)
                    logits = model(torch.randn(1, 3, 224, 224))
                    self.assertEqual(logits.shape, (1, 5))
                    if name == 'mobilenetv3':
                        self.assertEqual(model.num_features, 960)
                        self.assertEqual(model.embedding_dim, 1280)
                    F.cross_entropy(logits, torch.tensor([2])).backward()
                    # An unused old classifier or MobileNet projection would
                    # appear here as retained parameters with missing gradients.
                    for key, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)
                    model.eval()
                    with torch.inference_mode():
                        features = model.forward_features(torch.randn(1, 3, 192, 192))
                        self.assertEqual(model.forward_head(features).shape, (1, 5))

    def test_linear_head_and_nondefault_class_count(self):
        model = BackboneWrapper(Backbone, hidden_dims=(), num_classes=3)
        self.assertEqual(len(model.head), 1)
        self.assertEqual(model.head.classifier.in_features, 1280)
        self.assertEqual(model.head.classifier.out_features, 3)
        model.eval()
        with torch.inference_mode():
            self.assertEqual(model(torch.randn(1, 3, 224, 224)).shape, (1, 3))

    def test_activation_options_normalization_bias_and_backbone_options(self):
        model = BackboneWrapper(
            FDConvBackbone, hidden_dims=(24, 8), dropout=0.1, bias=False,
            activation=nn.LeakyReLU, activation_kwargs={'negative_slope': 0.2},
            normalization='batchnorm', backbone_kwargs={'in_chans': 1, 'kernel_num': 1},
        ).train()
        self.assertEqual(model.head.activation_0.negative_slope, 0.2)
        self.assertIsNot(model.head.activation_0, model.head.activation_1)
        self.assertIsInstance(model.head.norm_0, nn.BatchNorm1d)
        self.assertIsNone(model.head.classifier.bias)
        self.assertEqual(model.backbone.model.conv_stem.in_channels, 1)
        self.assertEqual(model.config['dropout'], [0.1, 0.1])
        logits = model(torch.randn(2, 1, 224, 224))
        self.assertEqual(logits.shape, (2, 5))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head.linear_0.weight.grad).all())

    def test_invalid_head_settings(self):
        for kwargs in (
            {'num_classes': 0}, {'hidden_dims': (0,)}, {'hidden_dims': (32.5,)},
            {'dropout': 1.0}, {'hidden_dims': (), 'dropout': -0.1},
            {'dropout': (0.1, 0.2)}, {'normalization': 'invalid'},
            {'backbone_kwargs': {'num_classes': 3}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BackboneWrapper(Backbone, **kwargs)
        with self.assertRaises(TypeError):
            BackboneWrapper(Backbone, activation=nn.ReLU())
        with self.assertRaises(TypeError):
            BackboneWrapper(nn.Identity())

    @unittest.skipUnless(torch.cuda.is_available(), 'Real Mamba wrapper integration requires CUDA')
    def test_mamba_and_hybrid_classes_with_custom_head(self):
        for backbone_class in (MambaBackbone, HybridBackbone):
            for name in Backbone.supported_names():
                with self.subTest(backbone=backbone_class.__name__, name=name):
                    model = BackboneWrapper(
                        backbone_class, name=name, hidden_dims=(32,),
                        backbone_kwargs={'hidden_dim': 32},
                    ).cuda().train()
                    logits = model(torch.randn(1, 3, 224, 224, device='cuda'))
                    self.assertEqual(logits.shape, (1, 5))
                    F.cross_entropy(logits, torch.tensor([2], device='cuda')).backward()
                    for key, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, key)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), key)


if __name__ == '__main__':
    unittest.main()
