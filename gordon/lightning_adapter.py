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

from parallel_adapter_strategy import (
    ParallelInputAdapterStrategy,
    ParallelInputAdapterStrategyConfig,
)

class LightningAdapterModule(pl.LightningModule, adapter_mixins.AdapterModuleMixin):
    """A LightningModule with NeMo adapter support."""

    def __init__(self, input_dim: int, base_model: SimpleRegressor):
        super().__init__()
        self.input_dim = input_dim
        self.base_model = base_model
        self.criterion = nn.MSELoss()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        with torch.no_grad():
            base_out = self.base_model(x)
        adapter_in = torch.cat([x, base_out], dim=1)
        if self.is_adapter_available():
            out = self.forward_enabled_adapters(adapter_in)
        else:
            out = adapter_in
        return out

    def training_step(
        self,
        batch: tuple[Float[Tensor, "batch 1"], Float[Tensor, "batch 1"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(
        self,
        batch: tuple[Float[Tensor, "batch 1"], Float[Tensor, "batch 1"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = []
        for name, param in self.named_parameters():
            if "adapter_layer" in name and param.requires_grad:
                params.append(param)
        return torch.optim.Adam(params, lr=0.01)

