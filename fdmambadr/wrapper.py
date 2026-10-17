"""Wrap a project backbone class with a configurable dense classifier.

Example::

    from torch import nn
    from fdmambadr.fdconv_models import FDConvBackbone
    from fdmambadr.wrapper import BackboneWrapper

    model = BackboneWrapper(
        FDConvBackbone,
        name="mobilenetv3",
        num_classes=5,
        hidden_dims=(512, 128),
        dropout=(0.3, 0.1),
        activation=nn.GELU,
        normalization="layernorm",
        backbone_kwargs={"kernel_num": 4},
    )
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from torch import Tensor, nn


class BackboneWrapper(nn.Module):
    """Construct a backbone class and replace its final linear classifier.

    ``backbone_class`` accepts ``Backbone``, ``FDConvBackbone``,
    ``MambaBackbone`` or ``HybridBackbone``. Pass the class, not an instance.
    The original pooling, MobileNet projection, FDConv and Mamba processing
    are retained. The original final classifier is removed from the model.

    Args:
        backbone_class: One of the project's backbone classes.
        name: Architecture name passed to the backbone constructor.
        num_classes: Number of output logits; no softmax is applied.
        hidden_dims: Widths of hidden dense layers; ``()`` gives a linear head.
        dropout: A probability shared by hidden layers, or one per hidden layer.
        activation: An ``nn.Module`` class, instantiated for each hidden layer.
        activation_kwargs: Constructor options such as LeakyReLU's negative_slope.
        normalization: ``None``, ``"layernorm"`` or ``"batchnorm"``. BatchNorm
            in a dense head requires more than one sample in training mode.
        bias: Whether hidden and output linear layers include biases.
        backbone_kwargs: Other constructor options, e.g. ``in_chans``,
            ``layer_path`` or Mamba's ``hidden_dim``. Set name/num_classes above.

    Hidden layers use Linear -> normalization -> activation -> dropout.
    ``forward_features`` returns spatial backbone features; ``forward_head``
    applies the original pooling/projection and the new head. All retained
    parameters remain trainable. Use identical head settings across ablations.
    """

    def __init__(
        self,
        backbone_class: type[nn.Module],
        name: str = "mobilenetv3",
        num_classes: int = 5,
        *,
        hidden_dims: Sequence[int] = (256,),
        dropout: float | Sequence[float] = 0.2,
        activation: type[nn.Module] = nn.ReLU,
        activation_kwargs: Mapping[str, Any] | None = None,
        normalization: str | None = None,
        bias: bool = True,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone_class, type) or not issubclass(backbone_class, nn.Module):
            raise TypeError("backbone_class must be an nn.Module class, not an instance")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        hidden_dims = tuple(hidden_dims)
        if any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in hidden_dims
        ):
            raise ValueError("hidden_dims must contain positive integer widths")
        if isinstance(dropout, Real):
            probabilities = (float(dropout),) * len(hidden_dims)
            # Validate a scalar even for a head without hidden layers.
            if not 0 <= dropout < 1:
                raise ValueError("dropout probabilities must be in [0, 1)")
        else:
            probabilities = tuple(dropout)
            if len(probabilities) != len(hidden_dims):
                raise ValueError("Provide one dropout probability per hidden layer")
        if any(not isinstance(p, Real) or not 0 <= p < 1 for p in probabilities):
            raise ValueError("dropout probabilities must be in [0, 1)")
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise TypeError("activation must be an nn.Module class, such as nn.ReLU")
        if normalization not in (None, "layernorm", "batchnorm"):
            raise ValueError("normalization must be None, 'layernorm' or 'batchnorm'")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")
        options = dict(backbone_kwargs or {})
        if {"name", "num_classes"}.intersection(options):
            raise ValueError("Set name and num_classes on the wrapper, not in backbone_kwargs")
        activation_options = dict(activation_kwargs or {})

        self.backbone = backbone_class(name=name, num_classes=num_classes, **options)
        if not all(
            callable(getattr(self.backbone, method, None))
            for method in ("forward_features", "forward_head")
        ):
            raise TypeError("The backbone must implement forward_features and forward_head")
        self.num_features = self.backbone.num_features
        self.num_classes = num_classes

        # All four project wrappers contain one timm model, at differing depths.
        # Keep it registered only through self.backbone to avoid duplicate state
        # entries. Its classifier's width includes MobileNet's conv_head output.
        native = next(
            (
                module
                for module in self.backbone.modules()
                if callable(getattr(module, "get_classifier", None))
                and callable(getattr(module, "reset_classifier", None))
            ),
            None,
        )
        if native is None or not isinstance(native.get_classifier(), nn.Linear):
            raise TypeError("Expected a project backbone containing a timm linear classifier")
        self.embedding_dim = native.get_classifier().in_features
        native.reset_classifier(0)

        layers: list[tuple[str, nn.Module]] = []
        input_dim = self.embedding_dim
        for index, (width, probability) in enumerate(zip(hidden_dims, probabilities)):
            layers.append((f"linear_{index}", nn.Linear(input_dim, width, bias=bias)))
            if normalization == "layernorm":
                layers.append((f"norm_{index}", nn.LayerNorm(width)))
            elif normalization == "batchnorm":
                layers.append((f"norm_{index}", nn.BatchNorm1d(width)))
            layers.append((f"activation_{index}", activation(**activation_options)))
            if probability:
                layers.append((f"dropout_{index}", nn.Dropout(float(probability))))
            input_dim = width
        layers.append(("classifier", nn.Linear(input_dim, num_classes, bias=bias)))
        self.head = nn.Sequential(OrderedDict(layers))
        self.config = dict(
            backbone_class=f"{backbone_class.__module__}.{backbone_class.__qualname__}",
            name=name,
            num_classes=num_classes,
            backbone_kwargs=options,
            hidden_dims=list(hidden_dims),
            dropout=list(probabilities),
            activation=f"{activation.__module__}.{activation.__qualname__}",
            activation_kwargs=activation_options,
            normalization=normalization,
            bias=bias,
            embedding_dim=self.embedding_dim,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        """Return spatial features, including any FDConv/Mamba processing."""
        return self.backbone.forward_features(x)

    def forward_head(self, features: Tensor) -> Tensor:
        """Pool/project spatial features, then return raw classification logits."""
        embedding = self.backbone.forward_head(features)
        return self.head(embedding)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))
