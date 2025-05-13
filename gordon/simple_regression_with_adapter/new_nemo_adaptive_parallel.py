import os
import time
from typing import Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from gordon.simple_regression_with_adapter.custom_adapter import (
    CustomAdapter,
)

# from gordon.simple_regression_with_adapter.lightning_adapter import LightningAdapterModule
from gordon.simple_regression_with_adapter.lightning_adapter import (
    LightningAdapterModule,
)
from gordon.simple_regression_with_adapter.model_configuration import (
    model_config,
)

# Load the trained base model
from gordon.simple_regression_with_adapter.nemo_lightning import (
    LossHistory,
    SimpleRegressor,
)
from gordon.simple_regression_with_adapter.parallel_adapter_strategy import (
    ParallelInputAdapterStrategy,
    ParallelInputAdapterStrategyConfig,
)

# END imports

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Get configuration from model_configuration.py
    cfg = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    print("Using model configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Extract parameters from config
    hidden_dim = cfg.model.hidden_dim

    # Load the base model with configured hidden_dim
    base_model = SimpleRegressor(hidden_dim=hidden_dim)
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

    # Create a custom adapter config for add_adapter
    adapter_config = OmegaConf.create(
        {
            "_target_": "gordon.simple_regression_with_adapter.custom_adapter.CustomAdapter",
            "size": 1,
            "hidden_dim": hidden_dim,
            "adapter_strategy": None,  # We'll set this to None initially
            "first_linear_bias": True,
            "second_linear_bias": True,
            "weight_init_method": "zeros",
            "activation_type": "tanh",
        }
    )

    # First add the adapter with a null strategy
    model.add_adapter("my_adapter", cfg=adapter_config)

    # Now create and set the strategy
    strategy_config = ParallelInputAdapterStrategyConfig(
        scaling_factor=1.0, in_features=2, out_features=1, bias=True
    )
    adapter_strategy = ParallelInputAdapterStrategy(
        scaling_factor=strategy_config.scaling_factor,
        in_features=strategy_config.in_features,
        out_features=strategy_config.out_features,
        bias=strategy_config.bias,
    )

    # Manually set the adapter_strategy for the adapter
    model.adapter_layer["my_adapter"].setup_adapter_strategy(adapter_strategy)

    # Enable the adapter
    model.set_enabled_adapters("my_adapter", enabled=True)
    model.adapter_layer["my_adapter"].adapter_unfreeze()

    # Make sure only adapter parameters are trainable
    for name, param in model.named_parameters():
        if "adapter_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    loss_history_adapter = LossHistory()

    trainer_adapter = pl.Trainer(
        max_epochs=20,
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
    print(f"{loss_history_adapter.train_losses[0:10]=}")
    print(f"{loss_history_adapter.val_losses[0:10]=}")
    plt.plot(loss_history_adapter.train_losses, label="Adapter Train Loss")
    plt.plot(loss_history_adapter.val_losses[1:], label="Adapter Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Adapter Loss Curves")
    plt.savefig("loss_adapter_lightning.png")
    plt.show()

    # Print the configuration parameters used
    print("Adapter Configuration Parameters:")
    print("-" * 30)
    print(f"Size: {adapter_config.size}")
    print(f"Hidden dimension: {adapter_config.hidden_dim}")
    print(f"First linear bias: {adapter_config.first_linear_bias}")
    print(f"Second linear bias: {adapter_config.second_linear_bias}")
    print(f"Weight init method: {adapter_config.weight_init_method}")
    print(f"Activation type: {adapter_config.activation_type}")

    # Strategy configuration
    print("\nAdapter Strategy Parameters:")
    print("-" * 30)
    print(f"Scaling factor: {strategy_config.scaling_factor}")
    print(f"In features: {strategy_config.in_features}")
    print(f"Out features: {strategy_config.out_features}")
    print(f"Bias: {strategy_config.bias}")
