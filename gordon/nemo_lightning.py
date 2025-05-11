import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float, Int  # Jaxtyping for type hints
from nemo.core import NeuralModule
from nemo.core.neural_types import AxisType, NeuralType, RegressionValuesType
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class SimpleRegressor(NeuralModule, pl.LightningModule):
    """A simple feedforward regressor using PyTorch Lightning.

    Args:
        hidden_dim: The number of hidden units in the hidden layer.

    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1).

        Returns:
            Output tensor of shape (batch, 1).

        """
        return self.net(x)

    def training_step(
        self,
        batch: tuple[Float[Tensor, "batch 1"], Float[Tensor, "batch 1"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Performs a training step.

        Args:
            batch: Tuple of input and target tensors.
            batch_idx: Index of the batch.

        Returns:
            The computed loss tensor.

        """
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
        """Performs a validation step.

        Args:
            batch: Tuple of input and target tensors.
            batch_idx: Index of the batch.

        Returns:
            The computed loss tensor.

        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer.

        Returns:
            The Adam optimizer.

        """
        return torch.optim.Adam(self.parameters(), lr=0.01)

    @property
    def input_types(self):
        return {"x": NeuralType(("B", "T"), RegressionValuesType())}

    @property
    def output_types(self):
        return {"y": NeuralType(("B", "T"), RegressionValuesType())}


# Callback to store losses for plotting
class LossHistory(pl.Callback):
    """Callback to store training and validation losses for plotting."""

    def __init__(self):
        super().__init__()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Stores the training loss at the end of each epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.

        """
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Stores the validation loss at the end of each epoch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being validated.

        """
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Set seed for reproducibility
    # pl.seed_everything(42, workers=True)

    # Data
    a = 1.0
    n = 1000
    m = 200

    data = np.load("sine_data.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = SimpleRegressor(hidden_dim=16)
    loss_history = LossHistory()

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=[loss_history],
    )

    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Save the trained model weights for later use (e.g., for adapters)
    torch.save(model.state_dict(), "base_model.pt")
    print("Base model checkpoint saved as 'base_model.pt'.")

    # Plot loss curves
    print(f"{loss_history.train_losses[0:5]=}")
    print(f"{loss_history.val_losses[0:5]=}")
    plt.plot(loss_history.train_losses, label="Train Loss")
    plt.plot(loss_history.val_losses[1:], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("loss_pure_lightning.png")
    plt.show()
