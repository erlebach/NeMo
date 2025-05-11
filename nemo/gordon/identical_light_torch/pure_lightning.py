import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import time

import numpy as np


class SimpleRegressor(pl.LightningModule):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f"{x=}, {y=}")
        y_hat = self(x)
        # print(f"{y_hat=}")
        loss = self.criterion(y_hat, y)
        print(f"{loss=}")
        self.log('train_loss', loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Callback to store losses for plotting
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

#----------------------------------------------------------------------
if __name__ == '__main__':
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Data
    a = 1.0
    n = 1000
    m = 200

    data = np.load('sine_data.npz')
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = SimpleRegressor(hidden_dim=16)
    for name, param in model.named_parameters():
        print(name, param.data)
    loss_history = LossHistory()
    
    trainer = pl.Trainer(
        max_epochs=10,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator='cpu',
        devices=1,
        callbacks=[loss_history]
    )
    
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")
    
    # Plot loss curves
    # print(f"{loss_history.train_losses[0:5]=}")
    # print(f"{loss_history.val_losses[0:5]=}")
    plt.plot(loss_history.train_losses, label='Train Loss')
    plt.plot(loss_history.val_losses[1:], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.savefig('loss_pure_lightning.png')
    plt.show()
