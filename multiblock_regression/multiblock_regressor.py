import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from nemo.core import NeuralModule
from nemo.core.neural_types import NeuralType, RegressionValuesType
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


class MultiBlockRegressor(NeuralModule, pl.LightningModule):
    """A multi-block regressor.

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

        self.criterion = nn.MSELoss()

    def forward(
        self, x: Float[Tensor, "batch input_dim"]
    ) -> Float[Tensor, "batch output_dim"]:
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

    def training_step(
        self,
        batch: tuple[
            Float[Tensor, "batch input_dim"], Float[Tensor, "batch output_dim"]
        ],
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
        batch: tuple[
            Float[Tensor, "batch input_dim"], Float[Tensor, "batch output_dim"]
        ],
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
    pl.seed_everything(42, workers=True)

    # Load data
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

    # Create model with multiple blocks
    model = MultiBlockRegressor(
        input_dim=1, hidden_dim=16, output_dim=1, num_blocks=2, num_layers_per_block=2
    )
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

    # Save the trained model weights for later use with adapters
    torch.save(model.state_dict(), "base_multiblock_model.pt")
    print("Base model checkpoint saved as 'base_multiblock_model.pt'.")

    # Plot loss curves
    plt.plot(loss_history.train_losses, label="Train Loss")
    plt.plot(loss_history.val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.savefig("multiblock_loss.png")
    plt.close()
