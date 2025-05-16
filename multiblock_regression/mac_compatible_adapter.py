from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from multiblock_regressor import MLP, MultiBlockRegressor, ResidualMLP


# Simple implementation of NeMo's AdapterWrapper
class AdapterWrapper(nn.Module):
    """Base class for adapter wrappers.

    This is a simplified version of NeMo's AdapterWrapper that works on Mac.
    It wraps a base module and adds adapter functionality to it.

    Args:
        to_wrap: The module to wrap with an adapter.
        adapter: The adapter module to apply.
    """

    def __init__(self, to_wrap: nn.Module, adapter: nn.Module):
        super().__init__()
        self.to_wrap = to_wrap
        self.adapter = adapter

    def forward(self, x):
        # Default implementation - should be overridden by subclasses
        return self.to_wrap(x)


# Simple implementation of NeMo's PEFT
class PEFT:
    """Base class for Parameter-Efficient Fine-Tuning methods.

    This is a simplified version of NeMo's PEFT that works on Mac.
    It provides an interface for transforming models with adapters.
    """

    def __init__(self):
        pass

    def transform(self, module: nn.Module, name=None, prefix=None):
        """Transform a module by adding adapters.

        Args:
            module: The module to transform.
            name: Optional name of the module.
            prefix: Optional prefix for the module path.

        Returns:
            The transformed module or the original module if no transformation is applied.
        """
        return module

    def __call__(self, model: nn.Module):
        """Apply the PEFT method to a model.

        Args:
            model: The model to transform.

        Returns:
            The transformed model.
        """
        return model


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
        config: Optional[Union[MultiBlockRegressorLoraConfig, Dict[str, Any]]] = None,
    ):
        super().__init__()

        # Convert config to the right format if needed
        if config is None:
            self.config = MultiBlockRegressorLoraConfig()
        elif isinstance(config, dict):
            self.config = MultiBlockRegressorLoraConfig(
                **{k: v for k, v in config.items()}
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

    def __call__(self, model: MultiBlockRegressor):
        """Apply LoRA adapters to the model.

        Args:
            model: MultiBlockRegressor model.

        Returns:
            The model with LoRA adapters applied.
        """
        if not isinstance(model, MultiBlockRegressor):
            raise ValueError(f"Expected MultiBlockRegressor, got {type(model)}")

        # Determine which blocks to adapt
        target_blocks = self.config.target_blocks
        if target_blocks is None:
            target_blocks = list(range(model.num_blocks))

        # Apply adapters to selected blocks
        for i in target_blocks:
            if i < 0 or i >= model.num_blocks:
                continue

            block_name = f"block_{i}"
            if not hasattr(model, block_name):
                continue

            block = getattr(model, block_name)
            wrapped_block = self.transform(block)

            # Replace the original block with the wrapped one
            setattr(model, block_name, wrapped_block)
            # Also update in the ModuleList
            model.blocks[i] = wrapped_block

        # Return the modified model
        return model


# Example of how to use MultiBlockRegressorLora
if __name__ == "__main__":
    import pytorch_lightning as pl
    from multiblock_regressor import LossHistory
    from torch.utils.data import DataLoader, TensorDataset

    # Load pre-trained model
    base_model = MultiBlockRegressor(
        input_dim=1, hidden_dim=16, output_dim=1, num_blocks=2, num_layers_per_block=2
    )
    base_model.load_state_dict(torch.load("base_multiblock_model.pt"))

    # Create LoRA configuration
    lora_config = MultiBlockRegressorLoraConfig(
        dim=8,
        alpha=32,
        dropout=0.1,
        target_blocks=[
            0,
            1,
        ],  # Apply to both blocks, could also be None to apply to all
    )

    # Create LoRA adapter and apply it to the model
    lora = MultiBlockRegressorLora(lora_config)
    adapted_model = lora(base_model)

    # Freeze base model weights and only train adapter weights
    for name, param in adapted_model.named_parameters():
        if "adapter" not in name and "lora_" not in name:
            param.requires_grad = False

    # Create data for fine-tuning (modified sine function)
    def create_modified_sine_data(n=1000, noise=0.1, phase=0.5, seed=42):
        np.random.seed(seed)
        x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)
        y = np.sin(x + phase) + noise * np.random.randn(*x.shape)
        return x, y

    x_finetune, y_finetune = create_modified_sine_data()
    x_val_finetune, y_val_finetune = create_modified_sine_data(
        n=200, noise=0.05, seed=43
    )

    train_dataset = TensorDataset(
        torch.tensor(x_finetune, dtype=torch.float32),
        torch.tensor(y_finetune, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val_finetune, dtype=torch.float32),
        torch.tensor(y_val_finetune, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Train the adapted model
    loss_history = LossHistory()

    # Create trainer and train
    trainer = pl.Trainer(
        max_epochs=5,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=[loss_history],
    )

    trainer.fit(
        adapted_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # Create plots to compare base model and adapted model
    with torch.no_grad():
        # Generate test data across the range
        x_test = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

        # Base model predictions
        base_model.eval()
        y_base = base_model(x_test_tensor).numpy()

        # Adapter model predictions
        adapted_model.eval()
        y_adapted = adapted_model(x_test_tensor).numpy()

        # Original sine
        y_true = np.sin(x_test)

        # Modified sine (target for adapter)
        y_modified = np.sin(x_test + 0.5)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true, "b-", label="Original Sine")
    plt.plot(x_test, y_modified, "g--", label="Modified Sine (Target)")
    plt.plot(x_test, y_base, "r-", label="Base Model")
    plt.plot(x_test, y_adapted, "m-", label="Adapted Model")
    plt.scatter(x_finetune, y_finetune, alpha=0.3, label="Fine-tuning Data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Base Model and LoRA-adapted Model")
    plt.savefig("lora_adaptation.png")
    plt.close()

    # Plot the loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history.train_losses, label="Train Loss")
    plt.plot(loss_history.val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LoRA Adaptation Loss")
    plt.savefig("lora_loss.png")
    plt.close()

    print("LoRA adaptation complete. Visualization saved to 'lora_adaptation.png'.")
