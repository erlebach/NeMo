import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.core import ModelPT
from omegaconf import DictConfig, OmegaConf
from typing import Union
from torch.utils.data import DataLoader, Dataset


# 1. Define a minimal dataset
class DummyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 1)  # Dummy input
        self.labels = torch.randn(size, 1)  # Dummy target

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 2. Define a minimal ModelPT subclass
class MinimalModelPT(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # ModelPT __init__ handles cfg and trainer setup
        super().__init__(cfg=cfg, trainer=trainer)

        # Minimal model layer
        self.linear = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward pass
        return self.linear(x)

    # --- LightningModule required methods ---
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)  # Log loss
        return loss

    def configure_optimizers(self):
        # Minimal optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        return optimizer

    # --- ModelPT data setup methods ---
    def setup_training_data(self, train_data_config: DictConfig | dict):
        # Setup training dataloader from config
        dataset = DummyDataset(size=train_data_config.dataset_size)
        self._train_dl = DataLoader(dataset, batch_size=train_data_config.batch_size)

    # Optional: Add validation and test setup if needed for trainer calls
    # def setup_validation_data(self, val_data_config): pass
    # def setup_test_data(self, test_data_config): pass

    # --- LightningModule data hooks (can delegate to _train_dl etc) ---
    def train_dataloader(self):
        return self._train_dl


# 3. Main execution block to test
if __name__ == "__main__":
    print("Testing MinimalModelPT with PyTorch Lightning Trainer...")

    # Create a dummy config needed by ModelPT
    dummy_cfg = OmegaConf.create(
        {
            "optim": {"lr": 0.01},
            "train_ds": {"dataset_size": 100, "batch_size": 10},
            # ModelPT also expects a 'model' section, though not used by MinimalModelPT
            "model": {},
        }
    )

    # Create a minimal trainer
    trainer = pl.Trainer(
        max_epochs=1,  # Run for 1 epoch
        accelerator="auto",  # Use available accelerator (cpu/gpu)
        devices=1,
        logger=False,  # Disable logging to keep output clean
        enable_checkpointing=False,  # Disable checkpointing
        enable_progress_bar=True,  # Show progress bar
    )

    # Instantiate the ModelPT subclass
    try:
        model = MinimalModelPT(cfg=dummy_cfg, trainer=trainer)
        print("MinimalModelPT instantiated successfully.")

        # Manually call setup_training_data as ModelPT might expect this
        # or rely on trainer.fit doing it based on config - let's try both
        # First, try letting trainer.fit handle it by passing config
        print("Calling trainer.fit...")
        trainer.fit(model, train_dataloaders=dummy_cfg.train_ds)
        print("trainer.fit completed successfully.")

        # You could also explicitly call setup_training_data if passing DataLoaders directly
        # model.setup_training_data(dummy_cfg.train_ds)
        # trainer.fit(model, train_dataloaders=model.train_dataloader())

    except TypeError as e:
        print(f"\nCaught a TypeError: {e}")
        print(
            "This suggests the Trainer does not recognize MinimalModelPT as a LightningModule."
        )
    except Exception as e:
        print(f"\nCaught an unexpected error: {e}")

    print("\nTest finished.")
