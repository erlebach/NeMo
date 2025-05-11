import time

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from nemo.collections.common.parts import adapter_modules
from nemo.core import adapter_mixins
from nemo_lightning import LossHistory, SimpleRegressor
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class LightningAdapterModule(pl.LightningModule, adapter_mixins.AdapterModuleMixin):
    """A LightningModule with NeMo adapter support.

    Args:
        input_dim: The input dimension (should match base model output + x).
        base_model: The frozen base model.

    """

    def __init__(self, input_dim: int, base_model: SimpleRegressor):
        super().__init__()
        self.input_dim = input_dim
        self.base_model = base_model
        self.criterion = nn.MSELoss()

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: Float[Tensor, "batch 1"]) -> Float[Tensor, "batch 1"]:
        """Forward pass through the base model and enabled adapters.

        Args:
            x: Input tensor of shape (batch, 1).

        Returns:
            Output tensor of shape (batch, 1).

        """
        with torch.no_grad():
            base_out = self.base_model(x)
        adapter_in = torch.cat([x, base_out], dim=1)  # shape: [batch, 2]
        # Pass through adapters if available
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
        # Only optimize adapter parameters
        params = []
        for name, param in self.named_parameters():
            if "adapter_layer" in name and param.requires_grad:
                params.append(param)
        return torch.optim.Adam(params, lr=0.01)


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

    # Set up LightningModule with NeMo adapter support
    model = LightningAdapterModule(input_dim=2, base_model=base_model)

    # Correct way: pass a config dict
    adapter_cfg = {
        "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
        "in_features": 2,
        "dim": 8,
        "activation": "relu",
    }
    model.add_adapter("my_adapter", cfg=adapter_cfg)
    model.set_enabled_adapters("my_adapter", enabled=True)
    model.adapter_layer["my_adapter"].adapter_unfreeze()

    # Make sure only adapter parameters are trainable
    for name, param in model.named_parameters():
        if "adapter_layer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

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


def test_lightning_adapter_module_forward() -> None:
    """Test the LightningAdapterModule forward pass.

    Returns:
        None

    """
    base_model = SimpleRegressor(hidden_dim=16)
    for param in base_model.parameters():
        param.requires_grad = False
    model = LightningAdapterModule(input_dim=2, base_model=base_model)
    # Add adapter
    adapter_cfg = adapter_modules.LinearAdapter(in_features=2, dim=8, activation="relu")
    model.add_adapter("my_adapter", cfg=adapter_cfg)
    model.set_enabled_adapters("my_adapter", enabled=True)
    model.adapter_layer["my_adapter"].adapter_unfreeze()
    x = torch.randn(4, 1)
    with torch.no_grad():
        base_out = base_model(x)
    adapter_in = torch.cat([x, base_out], dim=1)
    y = model.forward_enabled_adapters(adapter_in)
    assert y.shape == (4, 2) or y.shape == (4, 1)
    print(f"LightningAdapterModule forward test passed: {y.shape=}")


if __name__ == "__main__":
    test_lightning_adapter_module_forward()
