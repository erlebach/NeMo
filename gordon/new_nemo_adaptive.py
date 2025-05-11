import time
from dataclasses import dataclass

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
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from gordon.pure_lightning import SimpleRegressor


class MultiplicationAdapterStrategy(adapter_mixin_strategies.AbstractAdapterStrategy):
    """Adapter strategy that multiplies input and adapter output."""

    def __init__(self, scaling_factor: float = 1.0):
        super().__init__()
        self.scale = scaling_factor

    def forward(
        self,
        input: torch.Tensor,
        adapter: torch.nn.Module,
        *,
        module: "adapter_mixins.AdapterModuleMixin",
    ):
        adapter_out = adapter(input)
        return self.scale * (input * adapter_out)


@dataclass
class MultiplicationAdapterStrategyConfig:
    scaling_factor: float = 1.0
    _target_: str = f"{MultiplicationAdapterStrategy.__module__}.{MultiplicationAdapterStrategy.__name__}"


class SimpleRegressorAdapter(SimpleRegressor, AdapterModuleUtil):
    # class MultiplicativeAdapter(torch.nn.Module, AdapterModuleUtil):
    """A simple multiplicative adapter."""

    def __init__(
        self,
        size: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        adapter_strategy: MultiplicationAdapterStrategy = None,
    ):
        super().__init__()

        

        # Prepare the adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # Initialize the weights
        self.reset_parameters(self)

    def reset_parameters(self):
        # Initialize so that the adapter outputs ones at start (multiplicative identity)
        with torch.no_grad():
            self.model[0].weight = torch.nn.Parameter(torch.eye(self.size))
            self.model[-1].weight = torch.nn.Parameter(torch.eye(self.size))

    def get_default_strategy_config(self):
        return MultiplicationAdapterStrategyConfig()


if __name__ == "__main__":
    # Load the trained base model
    from nemo_lightning import LossHistory, SimpleRegressor

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

    # Instantiate the LightningModule
    model = LightningAdapterModule(input_dim=2, base_model=base_model)

    # Multiplicative adapter config with increased capacity
    adapter_cfg = OmegaConf.load("adapter_cfg.yaml")
    model.add_adapter("my_multiplicative_adapter", cfg=adapter_cfg)
    model.set_enabled_adapters("my_multiplicative_adapter", enabled=True)
    model.adapter_layer["my_multiplicative_adapter"].adapter_unfreeze()

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
