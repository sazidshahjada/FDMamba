"""Scratch-initialized FD-MambaDR classifiers; import variants as needed.

B0: fdmambadr.backbone.Backbone
B1: fdmambadr.fdconv_models.FDConvBackbone
B2: fdmambadr.mamba_models.MambaBackbone
B3: fdmambadr.hybrid_models.HybridBackbone
Custom dense heads: fdmambadr.wrapper.BackboneWrapper
"""


def build_model(*args, **kwargs):
    """Lazily construct a scratch-initialized B0/B1/B2/B3 classifier."""
    from .factory import build_model as build

    return build(*args, **kwargs)
