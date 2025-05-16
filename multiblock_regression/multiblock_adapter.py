from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd
from jaxtyping import Float
from multiblock_regression.multiblock_regressor import (
    MLP,
    MultiBlockRegressor,
    ResidualMLP,
)
from nemo.core import ModelPT, adapter_mixins
from nemo.core.classes.common import typecheck
from nemo.core.config import hydra_runner
from nemo.core.neural_types import NeuralType, RegressionValuesType
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


# Simplified PEFT base class - implements locally since we can't rely on NeMo's implementation
class PEFT:
    def __init__(self):
        pass

    def transform(self, module, name=None, prefix=None):
        """Transform a module. To be implemented by subclasses."""
        return module

    def __call__(self, model):
        """Apply transformations to model."""
        return model


# Simplified AdapterWrapper - implements locally
class AdapterWrapper(nn.Module):
    def __init__(self, to_wrap, adapter):
        super().__init__()
        self.base_module = to_wrap
        self.adapter = adapter


class LinearAdapter(nn.Module):
    """Low-rank adapter for linear layers.

    Args:
        in_features: Input features.
        out_features: Output features.
        dim: Dimension of the low-rank projection.
        alpha: Scaling factor for the adapter output.
        dropout: Dropout probability.
        bias: Whether to use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.scale = alpha / dim
        self.in_features = in_features
        self.out_features = out_features

        # Low-rank projection: in_features -> dim -> out_features
        self.lora_a = nn.Linear(in_features, dim, bias=False)
        self.lora_b = nn.Linear(dim, out_features, bias=bias)

        # Initialize LoRA weights - A with small random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Apply dropout
        x = self.dropout(x)

        # LoRA forward pass
        x = self.lora_a(x)  # Project to low-rank space
        x = self.lora_b(x)  # Project back to output space

        # Scale output
        return x * self.scale


class MLPAdapter(nn.Module):
    """Adapter for MLP modules.

    Args:
        in_features: Input features.
        hidden_features: Hidden features.
        out_features: Output features.
        dim: Adapter dimension.
        alpha: Adapter scaling factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1_adapter = LinearAdapter(
            in_features=in_features,
            out_features=hidden_features,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

        self.fc2_adapter = LinearAdapter(
            in_features=hidden_features,
            out_features=out_features,
            dim=dim,
            alpha=alpha,
            dropout=dropout,
        )

    def forward_fc1(self, x):
        return self.fc1_adapter(x)

    def forward_fc2(self, x):
        return self.fc2_adapter(x)


class MLPLoraWrapper(AdapterWrapper):
    """Wrapper around an MLP module to add LoRA functionality.

    Args:
        to_wrap: The base MLP module to wrap.
        adapter: The adapter module.
    """

    def __init__(
        self,
        to_wrap: MLP,
        adapter: MLPAdapter,
    ):
        super().__init__(to_wrap=to_wrap, adapter=adapter)
        self.base_module = to_wrap

    def forward(self, x):
        # Base forward
        x1 = self.base_module.fc1(x)
        x1 = self.base_module.activation(x1)

        # Add fc1 adapter output
        adapter_out1 = self.adapter.forward_fc1(x)
        x1 = x1 + adapter_out1

        # Continue with base forward
        x2 = self.base_module.fc2(x1)

        # Add fc2 adapter output
        adapter_out2 = self.adapter.forward_fc2(x1)
        x2 = x2 + adapter_out2

        return x2


class ResidualMLPAdapter(nn.Module):
    """Adapter for ResidualMLP modules.

    Args:
        dim: Feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        adapter_dim: Adapter dimension.
        alpha: Adapter scaling factor.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        adapter_dim: int = 8,
        alpha: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.alpha = alpha
        self.dropout = dropout

        # Create adapters for each layer
        self.adapters = nn.ModuleList(
            [
                MLPAdapter(
                    in_features=dim,
                    hidden_features=hidden_dim,
                    out_features=dim,
                    dim=adapter_dim,
                    alpha=alpha,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def get_adapter(self, idx: int):
        """Get adapter at the specified index.

        Args:
            idx: Adapter index.

        Returns:
            The adapter at the specified index.
        """
        if idx < 0 or idx >= len(self.adapters):
            raise IndexError(f"Adapter index {idx} out of range")
        return self.adapters[idx]


class ResidualMLPLoraWrapper(AdapterWrapper):
    """Wrapper around a ResidualMLP module to add LoRA functionality.

    Args:
        to_wrap: The base ResidualMLP module to wrap.
        adapter: The adapter module.
    """

    def __init__(
        self,
        to_wrap: ResidualMLP,
        adapter: ResidualMLPAdapter,
    ):
        super().__init__(to_wrap=to_wrap, adapter=adapter)
        self.base_module = to_wrap

        # Create wrapped versions of each layer
        self.adapter_layers = nn.ModuleList()
        for i, layer in enumerate(to_wrap.layers):
            # Get the corresponding adapter for this layer
            layer_adapter = adapter.get_adapter(i)
            # Create wrapped layer
            wrapped_layer = MLPLoraWrapper(to_wrap=layer, adapter=layer_adapter)
            self.adapter_layers.append(wrapped_layer)

    def forward(self, x):
        # Replicate the original forward but with wrapped layers
        input_x = x
        for layer in self.adapter_layers:
            x = layer(x)
            x = x + input_x  # Residual connection
            input_x = x

        return x


@dataclass
class MultiBlockRegressorLoraConfig:
    """Configuration for MultiBlockRegressorLora.

    Args:
        dim: Dimension of the low-rank projection.
        alpha: Scaling factor for the adapter output.
        dropout: Dropout probability.
        target_blocks: List of block indices to apply adapters to. If None, applies to all blocks.
    """

    dim: int = 8
    alpha: int = 32
    dropout: float = 0.0
    target_blocks: Optional[List[int]] = None


class MultiBlockRegressorLora(PEFT):
    """LoRA implementation for MultiBlockRegressor.

    This class implements the Parameter-Efficient Fine-Tuning (PEFT) interface
    for MultiBlockRegressor by adding LoRA adapters to specific blocks.

    Args:
        config: Configuration for LoRA adapters.
    """

    def __init__(
        self,
        config: Optional[
            Union[MultiBlockRegressorLoraConfig, DictConfig, Dict[str, Any]]
        ] = None,
    ):
        super().__init__()

        # Convert config to the right format if needed
        if config is None:
            self.config = MultiBlockRegressorLoraConfig()
        elif isinstance(config, dict):
            self.config = MultiBlockRegressorLoraConfig(
                **{k: v for k, v in config.items()}
            )
        elif isinstance(config, DictConfig):
            self.config = MultiBlockRegressorLoraConfig(
                **{k: v for k, v in OmegaConf.to_container(config).items()}
            )
        else:
            self.config = config

    def transform(self, m: nn.Module, name=None, prefix=None):
        """Apply LoRA transformation to modules.

        Args:
            m: The module to transform.
            name: The name of the module.
            prefix: The prefix path of the module.

        Returns:
            The transformed module if applicable, otherwise the original module.
        """
        if not isinstance(m, ResidualMLP):
            return m

        # Create a base adapter for this block
        adapter = ResidualMLPAdapter(
            dim=m.dim,
            hidden_dim=m.hidden_dim,
            num_layers=m.num_layers,
            adapter_dim=self.config.dim,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
        )

        return ResidualMLPLoraWrapper(to_wrap=m, adapter=adapter)

    def __call__(self, model: Union[MultiBlockRegressor, ModelPT]):
        """Apply LoRA adapters to the model.

        Args:
            model: MultiBlockRegressor or ModelPT model.

        Returns:
            The model with LoRA adapters applied.
        """
        # If ModelPT is provided, extract the actual MultiBlockRegressor
        if isinstance(model, ModelPT):
            if hasattr(model, "model"):
                actual_model = model.model
            else:
                actual_model = model
        else:
            actual_model = model

        if not isinstance(actual_model, MultiBlockRegressor):
            raise ValueError(f"Expected MultiBlockRegressor, got {type(actual_model)}")

        # Determine which blocks to adapt
        target_blocks = self.config.target_blocks
        if target_blocks is None:
            target_blocks = list(range(actual_model.num_blocks))

        # Apply adapters to selected blocks
        for i in target_blocks:
            if i < 0 or i >= actual_model.num_blocks:
                continue

            block_name = f"block_{i}"
            if not hasattr(actual_model, block_name):
                continue

            block = getattr(actual_model, block_name)
            wrapped_block = self.transform(block)

            # Replace the original block with the wrapped one
            setattr(actual_model, block_name, wrapped_block)
            # Also update in the ModuleList
            actual_model.blocks[i] = wrapped_block

        # Return the modified model
        return model


class MultiBlockRegressorModel(ModelPT, adapter_mixins.AdapterModelPTMixin):
    """MultiBlockRegressor model with NeMo ModelPT and adapter support.

    This class wraps the MultiBlockRegressor into NeMo's ModelPT framework,
    which allows it to be used with Hydra configuration and supports adapter-based
    fine-tuning.

    Args:
        cfg: Configuration object
        trainer: PyTorch Lightning trainer instance
    """

    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # Initialize parent classes
        super().__init__(cfg=cfg, trainer=trainer)

        # Create the base regressor model using the arch namespace instead of model
        self.regressor = MultiBlockRegressor(
            input_dim=cfg.arch.input_dim,
            hidden_dim=cfg.arch.hidden_dim,
            output_dim=cfg.arch.output_dim,
            num_blocks=cfg.arch.num_blocks,
            num_layers_per_block=cfg.arch.num_layers_per_block,
            activation=cfg.arch.activation,
        )

        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Setup adapters if needed
        self.setup_adapters()

        # Setup dataloaders if configs are available
        if hasattr(cfg, "model"):
            if hasattr(cfg.model, "train_ds"):
                self.setup_training_data(cfg.model.train_ds)
            if hasattr(cfg.model, "validation_ds"):
                self.setup_validation_data(cfg.model.validation_ds)
            if hasattr(cfg.model, "test_ds"):
                self.setup_test_data(cfg.model.test_ds)

    @property
    def input_types(self):
        return {"x": NeuralType(("B", "T"), RegressionValuesType())}

    @property
    def output_types(self):
        return {"y": NeuralType(("B", "T"), RegressionValuesType())}

    def forward(self, x):
        """Forward pass using the regressor"""
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        y_hat = self(x=x)  # Use keyword argument
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            The configured optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.get("weight_decay", 0.0),
        )

        # Configure learning rate scheduler if specified
        if hasattr(self.cfg.optim, "sched"):
            scheduler_config = self.cfg.optim.sched
            if scheduler_config.name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.T_max,
                    eta_min=scheduler_config.min_lr,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

        return optimizer

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        """Setup training data loader.

        Args:
            train_data_config: Configuration for training data
        """
        self._train_dl = self._get_dataloader_from_config(
            train_data_config, shuffle=True
        )

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        """Setup validation data loader.

        Args:
            val_data_config: Configuration for validation data
        """
        self._validation_dl = self._get_dataloader_from_config(
            val_data_config, shuffle=False
        )

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        """Setup test data loader.

        Args:
            test_data_config: Configuration for test data
        """
        self._test_dl = self._get_dataloader_from_config(
            test_data_config, shuffle=False
        )

    def _get_dataloader_from_config(
        self, config: Union[DictConfig, Dict], shuffle: bool = False
    ):
        """Helper method to create a dataloader from config.

        Args:
            config: Configuration for the dataloader
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader instance
        """
        import numpy as np

        # Load data from numpy file
        data = np.load(config.file_path)
        x = data["x"]
        y = data["y"]

        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )

        # Create dataloader
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False),
        )

    def add_adapter(self, name: str, cfg: Union[DictConfig, Dict]):
        """Add an adapter to the model.

        Args:
            name: Name of the adapter
            cfg: Configuration for the adapter
        """
        super().add_adapter(name, cfg)

        if name.startswith("lora"):
            # Apply LoRA adapter
            lora_config = MultiBlockRegressorLoraConfig(
                dim=cfg.dim,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
                target_blocks=cfg.get("target_blocks", None),
            )

            # Create and apply the LoRA adapter
            lora = MultiBlockRegressorLora(lora_config)
            lora(self.regressor)

    def get_enabled_adapters(self):
        """Get the list of enabled adapters.

        Returns:
            List of enabled adapter names
        """
        return super().get_enabled_adapters()

    def is_adapter_available(self) -> bool:
        """Check if any adapter is available.

        Returns:
            True if at least one adapter is available
        """
        return super().is_adapter_available()

    def set_enabled_adapters(self, name=None, enabled=True):
        """Enable or disable adapters.

        Args:
            name: Name of the adapter to enable/disable, or None for all adapters
            enabled: Whether to enable or disable the adapter(s)
        """
        super().set_enabled_adapters(name, enabled)

    @property
    def adapter_module_names(self) -> List[str]:
        """Get the list of adapter module names.

        Returns:
            List of module names that support adapters
        """
        module_names = [""]
        for i in range(self.regressor.num_blocks):
            module_names.append(f"block_{i}")
        return module_names

    @property
    def default_adapter_module_name(self) -> str:
        """Get the default adapter module name.

        Returns:
            Name of the default adapter module
        """
        return "block_0"

    def check_valid_model_with_adapter_support_(self):
        """Check if the model supports adapters."""
        pass

    @classmethod
    def list_available_models(cls):
        """
        This method is required by ModelPT and should return a list of
        pretrained model weights available for this class.

        Returns:
            List of pretrained models available for this class
        """
        return []

    def train_dataloader(self):
        """Return the training dataloader with better error handling."""
        if not hasattr(self, "_train_dl") or self._train_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "train_ds"):
                print(
                    f"Setting up training dataloader from {self.cfg.model.train_ds.file_path}"
                )
                self.setup_training_data(self.cfg.model.train_ds)
            else:
                raise ValueError("No training data configuration found in model config")

        if self._train_dl is None:
            raise ValueError("Failed to create training dataloader")

        return self._train_dl

    def val_dataloader(self):
        """Return the validation dataloader with better error handling."""
        if not hasattr(self, "_validation_dl") or self._validation_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "validation_ds"):
                print(
                    f"Setting up validation dataloader from {self.cfg.model.validation_ds.file_path}"
                )
                self.setup_validation_data(self.cfg.model.validation_ds)
            else:
                raise ValueError(
                    "No validation data configuration found in model config"
                )

        if self._validation_dl is None:
            raise ValueError("Failed to create validation dataloader")

        return self._validation_dl

    def test_dataloader(self):
        """Return the test dataloader with better error handling."""
        if not hasattr(self, "_test_dl") or self._test_dl is None:
            if hasattr(self.cfg, "model") and hasattr(self.cfg.model, "test_ds"):
                print(
                    f"Setting up test dataloader from {self.cfg.model.test_ds.file_path}"
                )
                self.setup_test_data(self.cfg.model.test_ds)
            else:
                raise ValueError("No test data configuration found in model config")

        if self._test_dl is None:
            raise ValueError("Failed to create test dataloader")

        return self._test_dl

    @classmethod
    def restore_from(
        cls,
        restore_path,
        trainer=None,
        override_config_path=None,
        map_location=None,
        strict=True,
    ):
        """Override restore_from to properly handle config and dataloaders after restoration."""
        # Call the parent restore_from method
        model = super().restore_from(
            restore_path=restore_path,
            trainer=trainer,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=strict,
        )

        # Ensure the model knows it's been restored and needs to check dataloaders
        model._dataloaders_initialized = False

        return model


@hydra_runner(config_path=".", config_name="multiblock_fixed_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if data needs to be prepared
    import os

    if not os.path.exists("data/sine_train.npz"):
        logging.info("Data files not found. Creating datasets...")
        prepare_data()

    # Make sure Trainer settings don't conflict with exp_manager
    if "logger" in cfg.trainer and cfg.trainer.logger:
        cfg.trainer.logger = False

    if "enable_checkpointing" in cfg.trainer and cfg.trainer.enable_checkpointing:
        cfg.trainer.enable_checkpointing = False

    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model from checkpoint
    model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.restore_path, trainer=trainer
    )

    # Debug print
    print(f"Model restored from: {cfg.model.restore_path}")

    # Set up dataloaders explicitly
    if hasattr(cfg.model, "train_ds"):
        print(f"Setting up training data from: {cfg.model.train_ds.file_path}")
        try:
            model.setup_training_data(cfg.model.train_ds)
            print("Training data setup successful")
        except Exception as e:
            print(f"Error setting up training data: {e}")

    if hasattr(cfg.model, "validation_ds"):
        print(f"Setting up validation data from: {cfg.model.validation_ds.file_path}")
        try:
            model.setup_validation_data(cfg.model.validation_ds)
            print("Validation data setup successful")
        except Exception as e:
            print(f"Error setting up validation data: {e}")

    if hasattr(cfg.model, "test_ds"):
        print(f"Setting up test data from: {cfg.model.test_ds.file_path}")
        try:
            model.setup_test_data(cfg.model.test_ds)
            print("Test data setup successful")
        except Exception as e:
            print(f"Error setting up test data: {e}")

    # Try to verify dataloaders are created
    print(f"Training dataloader: {model._train_dl is not None}")
    print(f"Validation dataloader: {model._validation_dl is not None}")
    print(f"Test dataloader: {model._test_dl is not None}")

    # Add and enable the adapter
    if hasattr(cfg, "adapter"):
        adapter_cfg = cfg.adapter
        model.add_adapter(adapter_cfg.name, adapter_cfg)
        model.set_enabled_adapters(adapter_cfg.name, enabled=True)

        # Freeze base model parameters
        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    # Training
    logging.info("Starting training...")
    trainer.fit(model)
    logging.info("Training completed.")

    # Testing
    if hasattr(cfg, "test_ds") and cfg.test_ds.file_path is not None:
        logging.info("Running testing...")
        trainer.test(model)

    # Save the model if a path is specified
    if hasattr(cfg, "nemo_path") and cfg.nemo_path is not None:
        model.save_to(cfg.nemo_path)
        logging.info(f"Model saved to {cfg.nemo_path}")
