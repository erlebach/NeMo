import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mac_compatible_adapter import (
    MultiBlockRegressorLora,
    MultiBlockRegressorLoraConfig,
)

# Import the MultiBlockRegressor and the adapter classes
from multiblock_regressor import MultiBlockRegressor
from torch.utils.data import DataLoader, TensorDataset


def create_sine_data(n=1000, noise=0.1, seed=42):
    """Create sine wave data with noise for regression.

    Args:
        n: Number of data points.
        noise: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of input and target tensors.
    """
    np.random.seed(seed)
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, noise, n)

    # Reshape for regression
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)

    return torch.tensor(x), torch.tensor(y)


def create_modified_sine_data(n=1000, noise=0.1, phase=0.5, seed=42):
    """Create modified sine wave data with a phase shift.

    Args:
        n: Number of data points.
        noise: Standard deviation of Gaussian noise.
        phase: Phase shift (in radians).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of input and target tensors.
    """
    np.random.seed(seed)
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x + phase) + np.random.normal(0, noise, n)

    # Reshape for regression
    x = x.reshape(-1, 1).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)

    return torch.tensor(x), torch.tensor(y)


def train_model(model, dataloader, num_epochs=100, learning_rate=0.01):
    """Train a model on the given data.

    Args:
        model: The model to train.
        dataloader: DataLoader with training data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for optimization.

    Returns:
        List of losses during training.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return losses


def plot_results(x, y_true, y_pred_base, y_pred_adapted, title="Regression Results"):
    """Plot the true data and predictions.

    Args:
        x: Input data.
        y_true: True target values.
        y_pred_base: Predictions from the base model.
        y_pred_adapted: Predictions from the adapted model.
        title: Plot title.
    """
    plt.figure(figsize=(12, 6))

    # Convert tensors to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    if isinstance(y_pred_base, torch.Tensor):
        y_pred_base = y_pred_base.detach().numpy()
    if isinstance(y_pred_adapted, torch.Tensor):
        y_pred_adapted = y_pred_adapted.detach().numpy()

    # Sort by x for proper line plots
    sort_idx = np.argsort(x.flatten())
    x = x[sort_idx]
    y_true = y_true[sort_idx]
    y_pred_base = y_pred_base[sort_idx]
    y_pred_adapted = y_pred_adapted[sort_idx]

    # Plot
    plt.scatter(x, y_true, s=5, alpha=0.5, label="True Data")
    plt.plot(x, y_pred_base, "r-", linewidth=2, label="Base Model")
    plt.plot(x, y_pred_adapted, "g-", linewidth=2, label="Adapted Model")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("adapter_results.png")
    plt.close()


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create original sine data for pre-training
    x_train, y_train = create_sine_data(n=1000, noise=0.1)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create a separate test set
    x_test, y_test = create_sine_data(n=200, noise=0.1, seed=100)

    # Create modified sine data for fine-tuning (with phase shift)
    x_finetune, y_finetune = create_modified_sine_data(n=500, noise=0.1, phase=0.5)
    finetune_dataset = TensorDataset(x_finetune, y_finetune)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=64, shuffle=True)

    # Create a test set for the modified task
    x_test_modified, y_test_modified = create_modified_sine_data(
        n=200, noise=0.1, phase=0.5, seed=100
    )

    # 1. Create and train the base model
    print("Training base model on standard sine data...")
    model = MultiBlockRegressor(
        input_dim=1,
        hidden_dim=16,
        output_dim=1,
        num_blocks=2,
        num_layers_per_block=2,
        activation="tanh",
    )

    base_model_losses = train_model(model, train_dataloader, num_epochs=100)
    print("Base model training complete!")

    # Evaluate the base model on both test sets
    model.eval()
    with torch.no_grad():
        y_pred_base = model(x_test)
        y_pred_base_on_modified = model(x_test_modified)

        base_mse = nn.MSELoss()(y_pred_base, y_test).item()
        base_mse_on_modified = nn.MSELoss()(
            y_pred_base_on_modified, y_test_modified
        ).item()

        print(f"Base model MSE on original task: {base_mse:.6f}")
        print(f"Base model MSE on modified task: {base_mse_on_modified:.6f}")

    # 2. Use LoRA for parameter-efficient fine-tuning
    print("\nApplying LoRA adapter to the model...")

    # Configure the adapter
    lora_config = MultiBlockRegressorLoraConfig(
        dim=8,  # Dimension of the low-rank projection
        alpha=16,  # Scaling factor
        dropout=0.1,
        target_blocks=None,  # Apply to all blocks (you could specify [0] or [1] to apply to specific blocks)
    )

    # Create and apply the LoRA adapter
    lora_adapter = MultiBlockRegressorLora(config=lora_config)
    adapted_model = lora_adapter(model)

    # Freeze the base model parameters and only train the adapter
    for param in adapted_model.parameters():
        param.requires_grad = False

    # Unfreeze only the adapter parameters
    for name, param in adapted_model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True

    # Count trainable parameters
    base_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(
        p.numel() for p in adapted_model.parameters() if p.requires_grad
    )

    print(f"Base model parameters: {base_params}")
    print(f"Adapter trainable parameters: {adapter_params}")
    print(f"Parameter efficiency: {adapter_params / base_params * 100:.2f}%")

    # 3. Fine-tune the adapted model on the modified task
    print("\nFine-tuning with adapter on modified sine data...")
    adapter_losses = train_model(
        adapted_model, finetune_dataloader, num_epochs=100, learning_rate=0.005
    )
    print("Adapter fine-tuning complete!")

    # Evaluate the adapted model on both test sets
    adapted_model.eval()
    with torch.no_grad():
        y_pred_adapted = adapted_model(x_test_modified)
        adapted_mse = nn.MSELoss()(y_pred_adapted, y_test_modified).item()
        print(f"Adapted model MSE on modified task: {adapted_mse:.6f}")

    # 4. Plot the results
    plot_results(
        x_test_modified,
        y_test_modified,
        y_pred_base_on_modified,
        y_pred_adapted,
        title="LoRA Adapter Fine-tuning Results",
    )

    print(f"\nResults visualization saved to adapter_results.png")


if __name__ == "__main__":
    main()
