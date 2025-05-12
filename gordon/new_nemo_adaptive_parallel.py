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

from lightning_adapter import LightningAdapterModule


class CustomAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        size: int = 1,
        hidden_dim: int = 32,
        adapter_strategy: ParallelInputAdapterStrategy = None,
    ):
        super().__init__()
        self.size = size
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(size, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, size)
        )

        # Prepare the adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # Initialize weights if needed
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.model[0].weight.fill_(0.0)
            self.model[-1].weight.fill_(0.0)

    def forward(self, x):
        return self.model(x)

    def get_default_strategy_config(self):
        return ParallelInputAdapterStrategyConfig()


class SimpleRegressorAdapter(SimpleRegressor, AdapterModuleUtil):
    # class MultiplicativeAdapter(torch.nn.Module, AdapterModuleUtil):
    """A simple multiplicative adapter."""

    def __init__(
        self,
        size: int,
        hidden_dim: int = 32,
        adapter_strategy: ParallelInputAdapterStrategy = None,
    ):
        super().__init__(hidden_dim=hidden_dim)
        self.size = size

        # Prepare the adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # Initialize the weights
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize so that the adapter outputs ones at start (multiplicative identity)
        with torch.no_grad():
            self.model[0].weight = torch.nn.Parameter(torch.eye(self.size))
            self.model[-1].weight = torch.nn.Parameter(torch.eye(self.size))

    def get_default_strategy_config(self):
        return ParallelInputAdapterStrategyConfig()

    def forward(self, x):
        # Don't call super().forward() which would call self.net again
        # Instead, just process x directly with whatever transformation needed
        return x  # Or some transformation of x


if __name__ == "__main__":
    base_model = SimpleRegressor(hidden_dim=16)
    base_model.load_state_dict(torch.load("base_model.pt"))
    for param in base_model.parameters():
        param.requires_grad = False

    # Load the original data
    data = np.load("sine_data.npz")
    x_train = data["x_train"]
    x_val = data["x_val"]

    # Prepare new targets: sin(x) + x
    y_train_adapter = np.sin(x_train) + x_train
    y_val_adapter = np.sin(x_val) + x_val

    train_dataset_adapter = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_adapter, dtype=torch.float32),
    )
    val_dataset_adapter = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val_adapter, dtype=torch.float32),
    )

    train_loader_adapter = DataLoader(
        train_dataset_adapter, batch_size=32, shuffle=False, num_workers=0
    )
    val_loader_adapter = DataLoader(
        val_dataset_adapter, batch_size=32, shuffle=False, num_workers=0
    )


    # Instantiate the LightningModule
    model = LightningAdapterModule(input_dim=2, base_model=base_model)

    # Parallel input adapter config with increased capacity
    adapter_cfg = OmegaConf.load("adapter_cfg.yaml")
    model.add_adapter("my_adapter", cfg=cast(DictConfig, adapter_cfg))
    model.set_enabled_adapters("my_adapter", enabled=True)
    model.adapter_layer["my_adapter"].adapter_unfreeze()

    # Make sure only adapter parameters are trainable
    for name, param in model.named_parameters():
        if "adapter_layer" in name:
            param.requires_grad = True
            print("TRUE")
        else:
            param.requires_grad = False

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    loss_history_adapter = LossHistory()

    trainer_adapter = pl.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=[loss_history_adapter],
    )

    start = time.time()
    trainer_adapter.fit(
        model,
        train_dataloaders=train_loader_adapter,
        val_dataloaders=val_loader_adapter,
    )
    end = time.time()
    print(f"Adapter training completed in {end - start:.2f} seconds")

    # Plot loss curves for adapter
    print(f"{loss_history_adapter.train_losses[0:5]=}")
    print(f"{loss_history_adapter.val_losses[0:5]=}")
    plt.plot(loss_history_adapter.train_losses, label="Adapter Train Loss")
    plt.plot(loss_history_adapter.val_losses[1:], label="Adapter Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Adapter Loss Curves")
    plt.savefig("loss_adapter_lightning.png")
    plt.show()
