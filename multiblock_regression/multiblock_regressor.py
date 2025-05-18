import json
import logging
import os
import tarfile
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from jaxtyping import Float
from nemo.core.neural_types import NeuralType, RegressionValuesType
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class MLP(torch.nn.Module):
    """MLP layer.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        activation: Activation function to use.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "tanh"
    ):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)

        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()

        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class ResidualMLP(torch.nn.Module):
    """Residual MLP module.

    Args:
        dim: Feature dimension.
        num_layers: Number of MLP layers.
        hidden_dim: Hidden dimension for each MLP layer.
        activation: Activation function to use.
    """

    def __init__(
        self, dim: int, num_layers: int, hidden_dim: int, activation: str = "tanh"
    ):
        super().__init__()

        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList(
            [MLP(dim, hidden_dim, dim, activation) for _ in range(num_layers)]
        )

    def forward(self, x):
        input_x = x
        for layer in self.layers:
            x = layer(x)
            x = x + input_x  # Residual connection
            input_x = x
        return x


class MultiBlockRegressor(torch.nn.Module):
    """A multi-block regressor (core architecture).

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for each block.
        output_dim: Output dimension.
        num_blocks: Number of residual blocks.
        num_layers_per_block: Number of MLP layers per block.
        activation: Activation function to use.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_blocks: int = 2,
        num_layers_per_block: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Create the blocks
        self.blocks = nn.ModuleList()

        # First block takes the input dimension
        first_block = ResidualMLP(
            dim=input_dim,
            num_layers=num_layers_per_block,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.blocks.append(first_block)
        setattr(self, f"block_0", first_block)

        # Subsequent blocks maintain the same dimension
        for i in range(1, num_blocks):
            block = ResidualMLP(
                dim=input_dim,
                num_layers=num_layers_per_block,
                hidden_dim=hidden_dim,
                activation=activation,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

        # Final output projection if needed
        if input_dim != output_dim:
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Output tensor of shape (batch, output_dim).
        """
        # Pass through each block
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.output_proj(x)

        return x


# Inherit directly from pl.LightningModule and implement methods needed
class MultiBlockRegressorPT(pl.LightningModule):
    """A multi-block regressor wrapped as a Lightning Module.

    Uses composition to contain a MultiBlockRegressor instance.
    """

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # First call LightningModule's init
        super().__init__()

        # Store config
        self.cfg = cfg

        # Now create the regressor as a member variable
        self.regressor = MultiBlockRegressor(
            input_dim=cfg.arch.input_dim,
            hidden_dim=cfg.arch.hidden_dim,
            output_dim=cfg.arch.output_dim,
            num_blocks=cfg.arch.num_blocks,
            num_layers_per_block=cfg.arch.num_layers_per_block,
            activation=cfg.arch.activation,
        )

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Create dataloaders
        self._train_dl = None
        self._validation_dl = None
        self._test_dl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate to the regressor's forward method
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        optimizer = torch.optim.Adam(
            self.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay
        )

        if (
            hasattr(optim_cfg, "sched")
            and OmegaConf.select(optim_cfg, "sched.name") is not None
        ):
            scheduler_cfg = optim_cfg.sched
            if scheduler_cfg.name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=scheduler_cfg.T_max, eta_min=scheduler_cfg.min_lr
                )
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            else:
                raise NotImplementedError(
                    f"Scheduler {scheduler_cfg.name} not implemented"
                )
        else:
            return optimizer

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )
        logging.info("Train DataLoader prepared")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )
        logging.info("Validation DataLoader prepared")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        self._test_dl = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )
        logging.info("Test DataLoader prepared")

    def _get_dataloader_from_config(
        self, config: Union[DictConfig, Dict], shuffle: bool = False
    ):
        if config is None:
            return None

        try:
            file_path = config.file_path
            batch_size = config.batch_size
            num_workers = config.get("num_workers", 0)
            pin_memory = config.get("pin_memory", False)

            print(f"Loading data from: {file_path}")
            data = np.load(file_path)
            x = data["x"]
            y = data["y"]

            dataset = torch.utils.data.TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            print(f"DataLoader created successfully for {file_path}.")
            return dataloader
        except Exception as e:
            print(f"Error creating dataloader from config: {e}")
            raise

    # Add methods needed for PyTorch Lightning
    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._validation_dl

    def test_dataloader(self):
        return self._test_dl

    # Simple save and restore methods to mimic ModelPT
    def save_to(self, save_path):
        state_dict = self.state_dict()
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")


class LossHistory(pl.Callback):
    """Callback to store training and validation losses for plotting."""

    def __init__(self):
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    dummy_cfg = OmegaConf.create(
        {
            "arch": {
                "input_dim": 1,
                "hidden_dim": 16,
                "output_dim": 1,
                "num_blocks": 2,
                "num_layers_per_block": 2,
                "activation": "tanh",
            },
            "optim": {
                "name": "adam",
                "lr": 0.01,
                "weight_decay": 0.0,
                "sched": {"name": "CosineAnnealing", "T_max": 10, "min_lr": 0.0001},
            },
            "train_ds": {
                "file_path": "data/base/sine_train.npz",
                "batch_size": 32,
                "num_workers": 0,
                "pin_memory": False,
            },
            "validation_ds": {
                "file_path": "data/base/sine_val.npz",
                "batch_size": 32,
                "num_workers": 0,
                "pin_memory": False,
            },
            "test_ds": {
                "file_path": "data/base/sine_test.npz",
                "batch_size": 32,
                "num_workers": 0,
                "pin_memory": False,
            },
        }
    )

    # Create the model
    model = MultiBlockRegressorPT(cfg=dummy_cfg)

    # Setup loss history callback
    loss_history = LossHistory()

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1,
        callbacks=[loss_history],
    )

    # Train the model
    start = time.time()
    # Data setup methods must be called before trainer.fit
    model.setup_training_data(dummy_cfg.train_ds)
    model.setup_validation_data(dummy_cfg.validation_ds)
    trainer.fit(model)
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the model
    model.save_to("base_multiblock_model.pt")
    print("Base model saved as 'base_multiblock_model.pt'.")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history.train_losses, label="Train Loss")
    plt.plot(loss_history.val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("multiblock_loss.png")
    plt.close()
    print("Loss curves plot saved as 'multiblock_loss.png'.")

    # Evaluate on test data
    print("\nEvaluating base model on test data...")
    # Data setup method must be called before trainer.test
    model.setup_test_data(dummy_cfg.test_ds)
    trainer.test(model)

    # Visualize test predictions
    try:
        test_data_for_plot = np.load(dummy_cfg.test_ds.file_path)
        x_test_plot = test_data_for_plot["x"]
        y_test_plot = test_data_for_plot["y"]

        model.eval()
        with torch.no_grad():
            x_test_tensor_plot = torch.tensor(x_test_plot, dtype=torch.float32)
            y_base_pred_plot = model(x_test_tensor_plot).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(
            x_test_plot,
            y_test_plot,
            "bo",
            markersize=3,
            alpha=0.5,
            label="Original Data (Target)",
        )
        plt.plot(x_test_plot, y_base_pred_plot, "r-", label="Base Model Prediction")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Base Model Evaluation on Original Sine Task")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("base_model_evaluation.png")
        plt.close()
        print("Base model evaluation plot saved as 'base_model_evaluation.png'.")

    except Exception as e:
        print(f"Error during base model evaluation plotting: {e}")
