import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.core import ModelPT
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

# Add basic logging configuration for visibility
logging.basicConfig(level=logging.INFO)


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


# 2. Define a minimal ModelPT subclass with required abstract methods implemented
class MinimalModelPT(ModelPT):
    # Expect cfg to contain only model-specific parameters (like optim)
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer = None):
        # Check instance type at the very start of init
        self._check_lightning_instance("_init_start")

        # Pass the model-specific config and trainer to ModelPT's init
        super().__init__(cfg=cfg, trainer=trainer)

        # Check instance type after super().__init__
        self._check_lightning_instance("_init_after_super")

        # Minimal model layer - Access optim config from self.cfg
        self.linear = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()

    def _check_lightning_instance(self, location: str):
        """Helper to check if the instance is a LightningModule and report."""
        is_lightning = isinstance(self, pl.LightningModule)
        logging.info(
            f"Check in {location}: is instance of pl.LightningModule? {is_lightning}"
        )
        return is_lightning

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check instance type at the start of forward
        self._check_lightning_instance("forward_start")
        # Simple forward pass
        return self.linear(x)

    # --- LightningModule required methods ---
    def training_step(self, batch, batch_idx):
        # Check instance type at the start of training_step
        self._check_lightning_instance("training_step_start")
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)  # Log loss
        return loss

    def configure_optimizers(self):
        # Check instance type at the start of configure_optimizers
        self._check_lightning_instance("configure_optimizers_start")
        # Access optim config from self.cfg
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr)
        return optimizer

    # --- ModelPT data setup methods (Required abstract methods) ---
    # These methods receive the *full* experiment config sections (e.g., cfg.train_ds)
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        # Check instance type at the start of setup_training_data
        self._check_lightning_instance("setup_training_data_start")
        if train_data_config is not None:
            dataset = DummyDataset(size=train_data_config.dataset_size)
            self._train_dl = DataLoader(
                dataset, batch_size=train_data_config.batch_size
            )
            logging.info("setup_training_data called and dataloader set.")
        else:
            self._train_dl = None
            logging.info("setup_training_data called with None config.")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict]):
        # Check instance type at the start of setup_validation_data
        self._check_lightning_instance("setup_validation_data_start")
        if val_data_config is not None:
            dataset = DummyDataset(size=val_data_config.dataset_size)
            self._validation_dl = DataLoader(
                dataset, batch_size=val_data_config.batch_size
            )
            logging.info("setup_validation_data called and dataloader set.")
        else:
            self._validation_dl = None
            logging.info("setup_validation_data called with None config.")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        # Check instance type at the start of setup_test_data
        self._check_lightning_instance("setup_test_data_start")
        if test_data_config is not None:
            dataset = DummyDataset(size=test_data_config.dataset_size)
            self._test_dl = DataLoader(dataset, batch_size=test_data_config.batch_size)
            logging.info("setup_test_data called and dataloader set.")
        else:
            self._test_dl = None
            logging.info("setup_test_data called with None config.")

    # --- LightningModule data hooks (delegates to _train_dl etc) ---
    def train_dataloader(self):
        # Check instance type at the start of train_dataloader
        self._check_lightning_instance("train_dataloader_start")
        return self._train_dl

    def val_dataloader(self):
        # Check instance type at the start of val_dataloader
        self._check_lightning_instance("val_dataloader_start")
        return self._validation_dl

    def test_dataloader(self):
        # Check instance type at the start of test_dataloader
        self._check_lightning_instance("test_dataloader_start")
        return self._test_dl

    # --- Other required abstract methods from ModelPT ---
    @classmethod
    def list_available_models(cls) -> Optional[List[Tuple[str, str]]]:
        # Check if cls is a LightningModule (should be True for the class itself)
        is_lightning = isinstance(cls, type) and issubclass(cls, pl.LightningModule)
        logging.info(
            f"Check in list_available_models (classmethod): is subclass of pl.LightningModule? {is_lightning}"
        )
        return None


# 3. Main execution block to test
if __name__ == "__main__":
    logging.info("Testing MinimalModelPT with PyTorch Lightning Trainer...")

    # Create the full experiment config structure
    full_experiment_cfg = OmegaConf.create(
        {
            "model": {  # Model specific config nested under 'model' key
                "optim": {"lr": 0.01},
                # Other model params would go here (e.g., arch)
            },
            "train_ds": {"dataset_size": 100, "batch_size": 10},
            "validation_ds": {"dataset_size": 50, "batch_size": 10},
            "test_ds": {"dataset_size": 50, "batch_size": 10},
            # Other experiment level configs (trainer, exp_manager) would be here
        }
    )

    # Create a minimal trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    try:
        # Worked
        # Call model's setup methods with the respective configurations FIRST.
        # The trainer, when fit/test is called without explicit dataloaders,
        # will then use the model's train_dataloader(), val_dataloader(), test_dataloader()
        # methods.
        logging.info("Calling model.setup_training_data manually...")
        model.setup_training_data(train_data_config=full_experiment_cfg.train_ds)
        logging.info("Calling model.setup_validation_data manually...")
        model.setup_validation_data(val_data_config=full_experiment_cfg.validation_ds)

        logging.info("Calling trainer.fit...")
        # trainer.fit will call model.train_dataloader() and model.val_dataloader()
        trainer.fit(model)
        logging.info("trainer.fit completed successfully.")

        logging.info("\nCalling model.setup_test_data manually...")
        model.setup_test_data(test_data_config=full_experiment_cfg.test_ds)

        logging.info("\nCalling trainer.test...")
        # trainer.test will call model.test_dataloader()
        trainer.test(model)
        logging.info("trainer.test completed successfully.")
    except Exception as e:
        logging.error(f"\nCaught in corrected attempt {e}")
        import traceback

    # Instantiate the ModelPT subclass with trainer=None
    # Crashed
    try:
        logging.info("Instantiating MinimalModelPT with trainer=None...")
        model = MinimalModelPT(cfg=full_experiment_cfg.model, trainer=None)
        logging.info("MinimalModelPT instantiated successfully.")

        # Check instance type immediately after instantiation
        model._check_lightning_instance("after_instantiation")

        # Set the trainer explicitly after instantiation
        logging.info("Setting trainer using model.set_trainer...")
        model.set_trainer(trainer)
        logging.info("Trainer set successfully.")

        # Check instance type after setting trainer
        model._check_lightning_instance("after_setting_trainer")

        # Manually set up data using config sections
        logging.info("Manually setting up data...")
        model.setup_training_data(full_experiment_cfg.train_ds)
        model.setup_validation_data(full_experiment_cfg.validation_ds)
        model.setup_test_data(full_experiment_cfg.test_ds)
        logging.info("Data setup completed.")

        # Now call trainer.fit/test. Pass only the model instance.
        # Lightning will use the dataloader hooks, which call _check_lightning_instance inside.
        logging.info("Calling trainer.fit...")
        trainer.fit(model)
        logging.info("trainer.fit completed successfully.")

        logging.info("\nCalling trainer.test...")
        trainer.test(model)
        logging.info("trainer.test completed successfully.")

    except TypeError as e:
        logging.error(f"\nCaught a TypeError: {e}")
        logging.error("This suggests an issue with type checking.")
        import traceback

        traceback.print_exc()
    except ValueError as e:
        logging.error(f"\nCaught a ValueError: {e}")
        logging.error("This suggests a configuration structure issue.")
        import traceback

        traceback.print_exc()
    except Exception as e:
        logging.error(f"\nCaught an unexpected error: {e}")
        import traceback

        traceback.print_exc()

    logging.info("\nTest finished.")
