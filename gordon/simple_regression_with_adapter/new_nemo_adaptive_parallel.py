import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

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
    # ==> Get configuration from model_configuration.py

    cfg = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    print("Using model configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Extract parameters from config
    hidden_dim = cfg.model.hidden_dim

    # Load the base model with configured hidden_dim
    # I'd like to make sure the loaded model is consistent with SimpleRegressor.
    # What is hidden_dim in the saved model is 16 and base_model was
    # initialized with hidden_dim = 32? What is the accepted approach?  Ideally, I'd like hidden_dim
    # to be overritten with the value from based_model.  How are mismatches avoided when everything is controlled
    # via configuration files?  can I execute torch.load() from outside base model?
    base_model = SimpleRegressor(hidden_dim=hidden_dim)
    try:
        base_model.load_state_dict(torch.load("base_model.pt"))
    except Exception as e:
        print("Model parameter mismatch when loading.")

    # What is hidden_dim now, of SimpleRegression? o
    for param in base_model.parameters():
        print(f"SimpleRegressor, {param.shape=}")
        param.requires_grad = False

    # ==> Handle data

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

    # Use the custom adapter config from model_configuration.py
    custom_adapter_config = cfg.model.custom_adapter
    # No need to set adapter_strategy to None anymore
    # custom_adapter_config.adapter_strategy = None  # <-- Remove this line

    # Setup LightningAdapterModule
    model = LightningAdapterModule(input_dim=2, base_model=base_model)
    model.add_adapter("my_adapter", cfg=custom_adapter_config)
    print(f"Model: {model}")

    # No need to create and set the strategy manually anymore
    # Remove these lines:
    # strategy_config = ParallelInputAdapterStrategyConfig(...)
    # adapter_strategy = ParallelInputAdapterStrategy(...)
    # model.adapter_layer["my_adapter"].setup_adapter_strategy(adapter_strategy)

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
    # print(f"Size: {adapter_config.size}")
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
