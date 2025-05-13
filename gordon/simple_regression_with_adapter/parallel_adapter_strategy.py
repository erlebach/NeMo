import time
from dataclasses import dataclass
from typing import cast

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

    def __init__(self, scaling_factor: float = 1.0):
        super().__init__()
        self.scale = scaling_factor
        self.linear = torch.nn.Linear(2, 1, bias=True)

    def forward(
        self,
        in_out: torch.Tensor,
        # output: torch.Tensor,
        adapter: torch.nn.Module,
        *,
        module: "AdapterModuleMixin",
    ):
        """
        Apply the adapter to the input and return the modified input.
        The base model will then process this modified input.

        Args:
            in_out: The original input+output related to the base model's forward() function
                They have the same shape, torch.cat along the 1st dimension
            adapter: The adapter module
            module: The parent module

        Returns:
            Modified input that will be passed to the base model
        """
        input, output = torch.split(in_out, in_out.shape[1] // 2, dim=1)
        adapter_output = adapter(input)
        result = self.linear(torch.cat([output, adapter_output], dim=1))
        return result


@dataclass
class ParallelInputAdapterStrategyConfig:
    scaling_factor: float = 1.0
    module = f"{ParallelInputAdapterStrategy.__module__}"
    name = f"{ParallelInputAdapterStrategy.__name__}"
    _target_: str = f"{module}.{name}"


