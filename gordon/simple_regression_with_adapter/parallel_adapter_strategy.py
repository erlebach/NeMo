import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from nemo.collections.common.parts import adapter_modules
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from nemo.core import adapter_mixins
from nemo.core.classes.mixins import adapter_mixin_strategies

# Load the trained base model
from nemo_lightning import LossHistory, SimpleRegressor
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class ParallelInputAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
    """
    An adapter strategy that applies the adapter to the input before the base model processes it.
    """

    def __init__(
        self,
        scaling_factor: float = 1.0,
        in_features: int = 2,
        out_features: int = 1,
        bias: bool = True,
    ):
        """Initialize the parallel input adapter strategy.

        Args:
            scaling_factor: Scaling factor for the adapter output.
            in_features: Number of input features for the linear layer.
            out_features: Number of output features for the linear layer.
            bias: Whether to include bias in the linear layer.
        """
        super().__init__()
        self.scale = scaling_factor
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(
        self,
        in_out: torch.Tensor,
        adapter: torch.nn.Module,
        *,
        module: "AdapterModuleMixin",
    ):
        """
        Apply the adapter to the input and return the modified input.

        Args:
            in_out: The original input+output tensor
            adapter: The adapter module
            module: The parent module

        Returns:
            Modified tensor
        """
        input, output = torch.split(in_out, in_out.shape[1] // 2, dim=1)
        adapter_output = adapter(input)
        result = self.linear(torch.cat([output, adapter_output], dim=1))
        return result


@dataclass
class ParallelInputAdapterStrategyConfig:
    scaling_factor: float = 1.0
    in_features: int = 2
    out_features: int = 1
    bias: bool = True
    module = f"{ParallelInputAdapterStrategy.__module__}"
    name = f"{ParallelInputAdapterStrategy.__name__}"
    _target_: str = f"{module}.{name}"
