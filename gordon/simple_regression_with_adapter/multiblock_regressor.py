import pytorch_lightning as pl
import torch
import torch.nn as nn
from jaxtyping import Float
from nemo.collections.common.parts.adapter_modules import (
    AdapterModuleUtil,
    LinearAdapter,
)
from nemo.core import NeuralModule, adapter_mixins
from nemo.core.neural_types import NeuralType, RegressionValuesType
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


# -----
class MLP(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    """MLP layer with adapter support.

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

        # Initialize adapter-related attributes
        self.adapter_layer = torch.nn.ModuleDict()
        self.adapter_cfg = OmegaConf.create({})

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

        # Forward through adapters if available
        if self.is_adapter_available():
            x = self.forward_enabled_adapters(x)

        return x

    # Add a utility method to calculate number of parameters
    @property
    def num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num


# ------
class ResidualMLPAdapter(torch.nn.Module, adapter_mixins.AdapterModuleMixin):
    """Residual MLP module with adapter support.

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

        # Initialize adapter-related attributes
        self.adapter_layer = torch.nn.ModuleDict()
        self.adapter_cfg = OmegaConf.create({})

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

        # Forward through adapters if available at this level
        if self.is_adapter_available():
            x = self.forward_enabled_adapters(x)

        return x

    # Override adapter methods to dispatch to the MLP layers
    def add_adapter(self, name: str, cfg):
        # First initialize the local adapter_layer
        super().add_adapter(name, cfg)

        # Then call add_adapter on each MLP layer
        for layer in self.layers:
            layer.add_adapter(name, cfg)

    def get_enabled_adapters(self):
        # Get local adapters
        enabled_adapters = super().get_enabled_adapters()

        # Collect enabled adapters from all layers
        for layer in self.layers:
            names = layer.get_enabled_adapters()
            for name in names:
                if name not in enabled_adapters:
                    enabled_adapters.append(name)

        return enabled_adapters

    def set_enabled_adapters(self, name=None, enabled=True):
        # Set state for local adapters
        super().set_enabled_adapters(name, enabled)

        # Set adapter state for all layers
        for layer in self.layers:
            layer.set_enabled_adapters(name, enabled)

    def is_adapter_available(self):
        # Check local adapters
        local_available = super().is_adapter_available()

        # Check if any layer has adapters
        layers_available = any([layer.is_adapter_available() for layer in self.layers])

        return local_available or layers_available


# Register the adapter for ResidualMLPAdapter
if adapter_mixins.get_registered_adapter(ResidualMLPAdapter) is None:
    adapter_mixins.register_adapter(ResidualMLPAdapter, ResidualMLPAdapter)


# ------
class MultiBlockRegressorAdapter(adapter_mixins.AdapterModelPTMixin):
    """Mixin to handle adapter functionality for MultiBlockRegressor."""

    def setup_adapters(self):
        # Check if modules support adapters
        supports_adapters = False

        for i in range(len(self.blocks)):
            block_name = f"block_{i}"
            if hasattr(self, block_name) and isinstance(
                getattr(self, block_name), adapter_mixins.AdapterModuleMixin
            ):
                supports_adapters |= True

        if supports_adapters:
            super().setup_adapters()

    def add_adapter(self, name: str, cfg):
        # Setup config
        super().add_adapter(name, cfg)

        # Resolve module name and adapter name
        module_name, adapter_name = self.resolve_adapter_module_name_(name)

        # Get global config
        global_config = self._get_global_cfg()

        # Forward to individual blocks based on module name
        if module_name == "":  # Global adapter
            for i in range(len(self.blocks)):
                block_name = f"block_{i}"
                if hasattr(self, block_name):
                    getattr(self, block_name).add_adapter(name, cfg)
        else:
            # Support specific block adapters with format "block_N:adapter_name"
            if module_name.startswith("block_"):
                block_idx = int(module_name.split("_")[1])
                if 0 <= block_idx < len(self.blocks):
                    getattr(self, f"block_{block_idx}").add_adapter(name, cfg)

    def get_enabled_adapters(self):
        enabled_adapters = super().get_enabled_adapters()

        # Collect from all blocks
        for i in range(len(self.blocks)):
            block_name = f"block_{i}"
            if hasattr(self, block_name) and isinstance(
                getattr(self, block_name), adapter_mixins.AdapterModuleMixin
            ):
                block_adapters = getattr(self, block_name).get_enabled_adapters()
                enabled_adapters.extend(block_adapters)

        return enabled_adapters

    def set_enabled_adapters(self, name=None, enabled=True):
        super().set_enabled_adapters(name, enabled)

        # Handle name resolution for specific blocks
        module_name = None
        if name is not None:
            module_name, _ = self.resolve_adapter_module_name_(name)

        # Forward to individual blocks
        for i in range(len(self.blocks)):
            block_name = f"block_{i}"
            if hasattr(self, block_name):
                block = getattr(self, block_name)
                if block.is_adapter_available():
                    # If name is None or this is the target block or it's a global adapter
                    if name is None or module_name == "" or module_name == block_name:
                        block.set_enabled_adapters(name, enabled)

    def _get_global_cfg(self):
        global_config = {}
        if (
            hasattr(self, "adapter_cfg")
            and self.adapter_global_cfg_key in self.adapter_cfg
        ):
            global_config = self.adapter_cfg[self.adapter_global_cfg_key]
        return global_config

    @property
    def adapter_module_names(self):
        module_names = super().adapter_module_names  # Default: ['']

        # Add all block names
        for i in range(len(self.blocks)):
            module_names.append(f"block_{i}")

        return module_names

    def check_valid_model_with_adapter_support_(self):
        # Add any additional validation here
        pass


# -----
class MultiBlockRegressor(NeuralModule, pl.LightningModule, MultiBlockRegressorAdapter):
    """A multi-block regressor with adapter support.

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
        cfg=None,
    ):
        # Initialize cfg as an OmegaConf DictConfig if not provided
        if cfg is None:
            self.cfg = OmegaConf.create({})
        else:
            self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

        # Initialize adapter_cfg as an OmegaConf DictConfig
        self.adapter_cfg = OmegaConf.create({})

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks

        # Create the blocks
        self.blocks = nn.ModuleList()

        # First block takes the input dimension
        first_block = ResidualMLPAdapter(
            dim=input_dim,
            num_layers=num_layers_per_block,
            hidden_dim=hidden_dim,
            activation=activation,
        )
        self.blocks.append(first_block)
        setattr(self, f"block_0", first_block)

        # Subsequent blocks maintain the same dimension
        for i in range(1, num_blocks):
            block = ResidualMLPAdapter(
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

        # Setup adapters
        self.setup_adapters()

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


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    import numpy as np
    from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
    from torch.utils.data import DataLoader, TensorDataset

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

    # Create and train the base model
    model = MultiBlockRegressor(
        input_dim=1, hidden_dim=16, output_dim=1, num_blocks=2, num_layers_per_block=2
    )

    # Create a callback to track loss
    class LossHistory(pl.Callback):
        def __init__(self):
            super().__init__()
            self.train_losses = []
            self.val_losses = []

        def on_train_epoch_end(self, trainer, pl_module):
            train_loss = trainer.callback_metrics.get("train_loss")
            if train_loss is not None:
                self.train_losses.append(train_loss.cpu().item())

        def on_validation_epoch_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None:
                self.val_losses.append(val_loss.cpu().item())

    loss_history = LossHistory()

    # Train the base model
    trainer = pl.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=[loss_history],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save base model weights
    torch.save(model.state_dict(), "base_multiblock_model.pt")

    # Freeze the base model and add adapters to specific blocks
    model.freeze()

    # Add adapters to all blocks
    adapter_cfg = LinearAdapterConfig(in_features=1, dim=8)
    model.add_adapter(name="global_adapter", cfg=adapter_cfg)

    # Add a block-specific adapter
    model.add_adapter(name="block_1:special_adapter", cfg=adapter_cfg)

    # Disable all adapters and enable only the one we want to train
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(name="block_1:special_adapter", enabled=True)

    # Unfreeze only the enabled adapters
    model.unfreeze_enabled_adapters()

    # Train only the adapters
    adapter_loss_history = LossHistory()
    adapter_trainer = pl.Trainer(
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=[adapter_loss_history],
    )

    adapter_trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # Save just the adapters
    model.save_adapters("multiblock_adapters.pt")

    # Example of loading adapters to a new model
    new_model = MultiBlockRegressor(
        input_dim=1, hidden_dim=16, output_dim=1, num_blocks=2, num_layers_per_block=2
    )

    # Load base model weights
    new_model.load_state_dict(torch.load("base_multiblock_model.pt"))

    # Load adapters
    new_model.load_adapters("multiblock_adapters.pt")

    # Enable specific adapter
    new_model.set_enabled_adapters(enabled=False)
    new_model.set_enabled_adapters(name="block_1:special_adapter", enabled=True)

    # Use the model with adapters
    x_test = torch.randn(1, 1)
    prediction = new_model(x_test)
    print(f"Prediction with adapter: {prediction.item()}")
