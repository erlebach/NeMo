import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    g = torch.Generator()
    g.manual_seed(42)

    data = np.load('sine_data.npz')
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']

    # Generate synthetic data
    a = 1.0
    n = 1000  # train samples
    m = 200   # val samples

    x_train = np.random.uniform(-5, 5, size=(n, 1))
    y_train = a * np.sin(x_train)
    x_val = np.random.uniform(-5, 5, size=(m, 1))
    y_val = a * np.sin(x_val)

    # Ensure y is 2D (N, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Define the model
    class SimpleRegressor(nn.Module):
        def __init__(self, hidden_dim: int = 16):
            """Simple feedforward regressor.

            Args:
                hidden_dim: Number of hidden units.

            """
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, 1).

            Returns:
                Output tensor of shape (batch, 1).

            """
            return self.net(x)

    # Instantiate model, loss, optimizer
    model = SimpleRegressor(hidden_dim=16)
    for name, param in model.named_parameters():
        print(name, param.data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 10
    train_losses = []
    val_losses = []

    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            # print(f"{xb=}, {yb=}")
            optimizer.zero_grad()
            pred = model(xb)
            # print(f"{pred=}")
            loss = criterion(pred, yb)
            print(f"{loss=}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    # Plot loss curves
    print(f"{train_losses[0:5]=}")
    print(f"{val_losses[0:5]=}")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig("loss_pure_torch.png")
    plt.show()

    # Self-contained test
    test_x = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    test_y = model(test_x)
    assert test_y.shape == (2, 1)
    print("Test passed: Model output shape is correct.")

if __name__ == "__main__":
    main()
