import os
import time
from typing import Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.collections.common.parts.adapter_modules import AdapterModuleUtil
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# from gordon.simple_regression_with_adapter.lightning_adapter import LightningAdapterModule
from gordon.simple_regression_with_adapter.lightning_adapter import (
    LightningAdapterModule,
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


# In new_nemo_adaptive_parallel.py - change this part
def load_adapter_config():
    # Load config template
    adapter_cfg_template = OmegaConf.load("adapter_template.yaml")

    # Inject the current module path at runtime
    current_module_path = "gordon.simple_regression_with_adapter"
    adapter_cfg = OmegaConf.create({"module_path": current_module_path})
    # Merge configs
    final_cfg = OmegaConf.merge(adapter_cfg, adapter_cfg_template)
    return final_cfg


# # Load config template
# adapter_cfg_template = OmegaConf.load("adapter_template.yaml")

# # Inject the current module path at runtime
# current_module_path = "gordon.simple_regression_with_adapter"  # Could get dynamically
# adapter_cfg = OmegaConf.create(
#     {"module_path": current_module_path, **adapter_cfg_template}
# )
# print(adapter_cfg)


class CustomAdapter(nn.Module, AdapterModuleUtil):
    def __init__(
        self,
        size: int = 1,
        hidden_dim: int = 32,
        adapter_strategy: ParallelInputAdapterStrategy = None,
        first_linear_bias: bool = True,
        second_linear_bias: bool = True,
        weight_init_method: str = "zeros",
        activation_type: str = "tanh",
    ):
        """Initialize a custom adapter.

        Args:
            size: Input and output size of the adapter.
            hidden_dim: Hidden dimension of the adapter.
            adapter_strategy: Strategy for applying the adapter.
            first_linear_bias: Whether to include bias in first linear layer.
            second_linear_bias: Whether to include bias in second linear layer.
            weight_init_method: Method for weight initialization ("zeros", "eye", or "normal").
            activation_type: Type of activation function ("tanh", "relu", or "sigmoid").
        """
        super().__init__()
        self.size = size
        self.hidden_dim = hidden_dim

        # Create activation function
        if activation_type.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_type.lower() == "relu":
            activation = nn.ReLU()
        elif activation_type.lower() == "sigmoid":
            activation = nn.Sigmoid()
        else:
            activation = nn.Tanh()

        # Create the model with configurations
        self.model = nn.Sequential(
            nn.Linear(size, hidden_dim, bias=first_linear_bias),
            activation,
            nn.Linear(hidden_dim, size, bias=second_linear_bias),
        )

        # Prepare the adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # Initialize weights based on method
        if weight_init_method.lower() == "zeros":
            self.reset_parameters()
        elif weight_init_method.lower() == "eye":
            self.reset_parameters_eye()
        elif weight_init_method.lower() == "normal":
            self.reset_parameters_normal()
        else:
            self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.model[0].weight.fill_(0.0)
            self.model[-1].weight.fill_(0.0)

    def reset_parameters_eye(self):
        with torch.no_grad():
            if self.model[0].weight.shape[0] == self.model[0].weight.shape[1]:
                self.model[0].weight = torch.nn.Parameter(
                    torch.eye(self.model[0].weight.shape[0])
                )
            if self.model[-1].weight.shape[0] == self.model[-1].weight.shape[1]:
                self.model[-1].weight = torch.nn.Parameter(
                    torch.eye(self.model[-1].weight.shape[0])
                )

    def reset_parameters_normal(self):
        with torch.no_grad():
            nn.init.normal_(self.model[0].weight, mean=0.0, std=0.01)
            nn.init.normal_(self.model[-1].weight, mean=0.0, std=0.01)

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


def read_adapter_config(
    config_path: str = "gordon/simple_regression_with_adapter/adapter_cfg.yaml",
) -> Dict[str, any]:
    """Read all configuration parameters from adapter_cfg.yaml.

    Args:
        config_path: Path to the adapter configuration file.

    Returns:
        Dictionary containing all the configuration parameters.

    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found")

    # Load the configuration
    config = OmegaConf.load(config_path)

    # Convert to dictionary for easier access
    config_dict = OmegaConf.to_container(config, resolve=True)

    return config_dict


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # What hidden dim is used? From argument or from adapter file?
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
    # adapter_cfg = load_adapter_config()
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

    # Read the configuration
    adapter_config = read_adapter_config()

    # Print the configuration parameters
    print("Adapter Configuration Parameters:")
    print("-" * 30)

    # Main adapter parameters
    print(f"Target class: {adapter_config['_target_']}")
    print(f"Size: {adapter_config['size']}")
    print(f"Hidden dimension: {adapter_config['hidden_dim']}")

    # Adapter strategy parameters
    print("\nAdapter Strategy Parameters:")
    print("-" * 30)
    print(f"Strategy class: {adapter_config['adapter_strategy']['_target_']}")
    print(f"Scaling factor: {adapter_config['adapter_strategy']['scaling_factor']}")

    # Access nested parameters
    strategy_config = adapter_config["adapter_strategy"]
    for key, value in strategy_config.items():
        if key != "_target_":
            print(f"{key}: {value}")
