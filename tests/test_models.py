"""Numerical regressions; run: python -m unittest discover -s tests -v.

CUDA tests use real Mamba and fail on broken/missing kernels when a GPU exists.
CPU skips do not count as verification of B2/B3.
"""
import gc
import importlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from fdmambadr.backbone import Backbone
from fdmambadr.FDConv import FDConv, FrequencyBandModulation
from fdmambadr.fdconv_models import FDConvBackbone, FP32FDConv, _compatible
from fdmambadr.mamba_models import LocalGlobalMambaFusion, MambaBackbone
from fdmambadr.hybrid_models import HybridBackbone


torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def assert_gradients(self, module):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_imports_do_not_require_optional_mamba_or_mmcv(self):
        for module in ('backbone', 'FDConv', 'fdconv_models', 'mamba_models', 'hybrid_models'):
            importlib.import_module(f'fdmambadr.{module}')

    def test_all_baselines_and_fdconv_variants_train_at_224(self):
        # Batch one catches FDConv's globally pooled BatchNorm regression.
        for constructor in (Backbone, FDConvBackbone):
            for name in Backbone.supported_names():
                with self.subTest(variant=constructor.__name__, backbone=name):
                    with patch('timm.create_model', wraps=importlib.import_module('timm').create_model) as factory:
                        model = constructor(name).train()
                    self.assertIs(factory.call_args.kwargs['pretrained'], False)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    images = torch.randn(1, 3, 224, 224, requires_grad=True)
                    features = model.forward_features(images)
                    self.assertEqual(features.shape[1], model.num_features)
                    logits = model.forward_head(features)
                    self.assertEqual(logits.shape, (1, 5))
                    loss = F.cross_entropy(logits, torch.tensor([2]))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assert_gradients(model)
                    self.assertTrue(torch.isfinite(images.grad).all())
                    if isinstance(model, FDConvBackbone):
                        layers = [m for m in model.modules() if isinstance(m, FDConv)]
                        self.assertEqual(len(layers), 1)
                        layer = layers[0]
                        self.assertTrue(layer.fdconv_active)
                    optimizer.step()
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                    # timm ResNet zero-initializes the last BN in each residual
                    # branch. Earlier convs receive nonzero gradients after
                    # that BN scale has received its first optimizer update.
                    optimizer.zero_grad(set_to_none=True)
                    second_loss = F.cross_entropy(model(images.detach()), torch.tensor([2]))
                    second_loss.backward()
                    self.assert_gradients(model)
                    if isinstance(model, FDConvBackbone):
                        self.assertGreater(layer.dft_weight.grad.abs().sum().item(), 0)
                    optimizer.step()
                    model.eval()
                    with torch.inference_mode():
                        self.assertEqual(model(images.detach()).shape, (1, 5))
                    del model, optimizer, images, features, logits, loss, second_loss
                    gc.collect()

    def test_fdconv_fft_shapes_and_single_kernel(self):
        for batch, cin, cout, kernel, kernels in (
            (1, 20, 24, 3, 4), (2, 21, 24, 3, 4),
            (1, 20, 24, 1, 1), (2, 21, 24, 1, 1),
        ):
            with self.subTest(batch=batch, channels=cin, kernel=kernel, kernels=kernels):
                layer = FDConv(cin, cout, kernel_size=kernel, padding=kernel // 2,
                               kernel_num=kernels, bias=True)
                x = torch.randn(batch, cin, 7, 9, requires_grad=True)
                y = layer(x)
                self.assertEqual(y.shape, (batch, cout, 7, 9))
                y.square().mean().backward()
                self.assert_gradients(layer)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(layer.dft_weight.grad.abs().sum().item(), 0)
                self.assertGreater(layer.KSM_Global.fc.weight.grad.abs().sum().item(), 0)
                self.assertGreater(layer.KSM_Local.conv.weight.grad.abs().sum().item(), 0)
                if hasattr(layer, 'FBM'):
                    self.assertGreater(layer.FBM.freq_weight_conv_list[0].weight.grad.abs().sum().item(), 0)

    def test_frequency_masks_use_actual_fft_bins(self):
        module = FrequencyBandModulation(20, k_list=(2, 4, 8))
        # Constant 2x attention makes output = 2*x - lowest-frequency(x).
        for conv in module.freq_weight_conv_list:
            nn.init.zeros_(conv.weight)
            nn.init.constant_(conv.bias, 1)
        module.sp_act = lambda weights: torch.full_like(weights, 2)
        for h, w in ((7, 9), (14, 14), (8, 8), (7, 8)):
            with self.subTest(shape=(h, w)):
                x = torch.randn(2, 20, h, w)
                frequencies = torch.maximum(torch.fft.fftfreq(h).abs()[:, None],
                                            torch.fft.rfftfreq(w).abs()[None, :])
                mask = frequencies < 0.5 / 8 + 1e-8
                lowest = torch.fft.irfft2(torch.fft.rfft2(x, norm='ortho') * mask,
                                         s=(h, w), norm='ortho')
                torch.testing.assert_close(module(x), 2 * x - lowest)

    def test_spatial_and_spectral_parameterizations_agree(self):
        torch.manual_seed(4)
        spatial = FDConv(20, 24, 3, padding=1, kernel_num=4, convert_param=False).eval()
        torch.manual_seed(4)
        spectral = FDConv(20, 24, 3, padding=1, kernel_num=4, convert_param=True).eval()
        x = torch.randn(2, 20, 7, 9)
        torch.testing.assert_close(spatial(x), spectral(x))

    def test_fp32_wrapper_under_cpu_autocast(self):
        layer = FP32FDConv(FDConv(20, 24, 3, padding=1))
        x = torch.randn(1, 20, 7, 9, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = layer(x)
            loss = y.float().square().mean()
        self.assertEqual(y.dtype, x.dtype)
        loss.backward()
        self.assert_gradients(layer)

    def test_fdconv_state_dict_round_trip(self):
        original = FDConv(20, 24, 3, padding=1).eval()
        x = torch.randn(1, 20, 7, 9)
        expected = original(x)
        checkpoint = io.BytesIO()
        torch.save(original.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = FDConv(20, 24, 3, padding=1).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(expected, restored(x))

    def test_fdconv_fallback_obeys_stride_eligibility(self):
        layer = FDConv(20, 24, 3, padding=1, stride=2)
        self.assertFalse(layer.fdconv_active)
        x = torch.randn(1, 20, 7, 9)
        torch.testing.assert_close(layer(x), F.conv2d(x, layer.weight, layer.bias, stride=2, padding=1))

    def test_invalid_fdconv_configurations_fail_early(self):
        for kwargs in ({'kernel_num': 0}, {'kernel_num': 7}, {'groups': 2},
                       {'param_reduction': 0}, {'spatial_freq_decompose': True},
                       {'ksm_global_act': 'softmax'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FDConv(20, 24, 3, padding=1, **kwargs)
        for kwargs in ({'use_fdconv_if_c_gt': 10000}, {'use_fdconv_if_k_in': ()},
                       {'convert_param': False}):
            with self.subTest(adapter_kwargs=kwargs), self.assertRaises(ValueError):
                FDConvBackbone('mobilenetv3', fdconv_kwargs=kwargs)

    def test_ineligible_replacement_sites_are_rejected(self):
        for path, conv in (
            ('blocks.2.se.conv_reduce', nn.Conv2d(64, 32, 1)),
            ('blocks.2.conv_dw', nn.Conv2d(32, 32, 3, groups=32)),
            ('conv_stem', nn.Conv2d(32, 32, 3)),
            ('blocks.2.conv', nn.Conv2d(32, 32, 3, padding_mode='reflect')),
        ):
            with self.subTest(path=path):
                self.assertFalse(_compatible(path, conv, max_channels=384))
        with self.assertRaises(ValueError):
            FDConvBackbone('resnet50', layer_path='missing')

    def test_hybrid_reuses_b2_fusion(self):
        from fdmambadr import hybrid_models, mamba_models
        self.assertIs(hybrid_models.LocalGlobalMambaFusion, mamba_models.LocalGlobalMambaFusion)

    @unittest.skipUnless(torch.cuda.is_available(), 'Real Mamba integration requires CUDA')
    def test_mamba_fusion_cuda_shape_and_gradients(self):
        for batch in (1, 2):
            with self.subTest(batch=batch):
                module = LocalGlobalMambaFusion(40, hidden_dim=32).cuda().train()
                x = torch.randn(batch, 40, 7, 9, device='cuda', requires_grad=True)
                y = module(x)
                self.assertEqual(y.shape, x.shape)
                y.square().mean().backward()
                self.assert_gradients(module)
                for branch in (module.local, module.mamba, module.gate):
                    self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in branch.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'Real Mamba integration requires CUDA')
    def test_all_mamba_and_hybrid_variants_cuda_with_amp(self):
        for constructor in (MambaBackbone, HybridBackbone):
            for name in Backbone.supported_names():
                for amp in (False, True):
                    with self.subTest(variant=constructor.__name__, backbone=name, amp=amp):
                        model = constructor(name, hidden_dim=32).cuda().train()
                        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                        scaler = torch.amp.GradScaler('cuda', enabled=amp)
                        x = torch.randn(1, 3, 224, 224, device='cuda')
                        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                            logits = model(x)
                            self.assertEqual(logits.shape, (1, 5))
                            loss = F.cross_entropy(logits, torch.tensor([2], device='cuda'))
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        self.assert_gradients(model)
                        scaler.step(optimizer)
                        scaler.update()
                        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        model.eval()
                        with torch.inference_mode():
                            self.assertEqual(model(x).shape, (1, 5))
                        del model, optimizer, scaler, x, logits, loss
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
