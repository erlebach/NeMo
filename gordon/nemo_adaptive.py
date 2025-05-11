import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from nemo_lightning import (  # Import from your previous file
    LossHistory,
    SimpleRegressor,
)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class Adapter(nn.Module):
    """A simple adapter module to be trained on top of a frozen base model.

    Args:
        input_dim: The input dimension (should match base model output + x).
        output_dim: The output dimension.

    """

    def __init__(self, input_dim: int = 2, output_dim: int = 1):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x: Float[Tensor, "batch 2"]) -> Float[Tensor, "batch 1"]:
        """Forward pass through the adapter.

        Args:
            x: Input tensor of shape (batch, 2).

        Returns:
            Output tensor of shape (batch, 1).

        """
        return self.adapter(x)


class AdapterRegressor(pl.LightningModule):
    """LightningModule for training an adapter on top of a frozen base model.

    Args:
        base_model: The frozen base model.
        adapter: The trainable adapter module.

    """

    def __init__(self, base_model: SimpleRegressor, adapter: Adapter):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.criterion = nn.MSELoss()
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        """Forward pass through the base model and adapter.

        Args:
            x: Input tensor of shape (batch, 1).

        Returns:
            Output tensor of shape (batch, 1).

        """
        with torch.no_grad():
            base_out = self.base_model(x)
        adapter_in = torch.cat([x, base_out], dim=1)  # shape: [batch, 2]
        return self.adapter(adapter_in)

    def training_step(
        self,
        batch: tuple[Float[Tensor, "batch 1"], Float[Tensor, "batch 1"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Performs a training step for the adapter.

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
        """Performs a validation step for the adapter.

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
        """Configures the optimizer for the adapter.

        Returns:
            The Adam optimizer.

        """
        return torch.optim.Adam(self.adapter.parameters(), lr=0.01)


if __name__ == "__main__":
    # Load the trained base model
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

    # Set up adapter and regressor
    adapter = Adapter(input_dim=2, output_dim=1)
    adapter_model = AdapterRegressor(base_model=base_model, adapter=adapter)
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
        adapter_model,
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


def test_adapter_forward() -> None:
    """Test the Adapter forward pass.

    Returns:
        None

    """
    adapter = Adapter(input_dim=2, output_dim=1)
    x = torch.randn(4, 2)
    y = adapter(x)
    assert y.shape == (4, 1)
    print(f"Adapter forward test passed: {y.shape=}")


def test_adapter_regressor_forward() -> None:
    """Test the AdapterRegressor forward pass.

    Returns:
        None

    """
    base_model = SimpleRegressor(hidden_dim=16)
    for param in base_model.parameters():
        param.requires_grad = False
    adapter = Adapter(input_dim=2, output_dim=1)
    model = AdapterRegressor(base_model, adapter)
    x = torch.randn(4, 1)
    y = model(x)
    assert y.shape == (4, 1)
    print(f"AdapterRegressor forward test passed: {y.shape=}")


if __name__ == "__main__":
    test_adapter_forward()
    test_adapter_regressor_forward()
