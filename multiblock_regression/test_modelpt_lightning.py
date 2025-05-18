import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nemo.core import ModelPT
from omegaconf import DictConfig, OmegaConf

# Add basic logging configuration for visibility
logging.basicConfig(level=logging.INFO)


# 2. Define a minimal ModelPT subclass with required abstract methods implemented
class MinimalModelPT(ModelPT):
    # Expect cfg to contain only model-specific parameters (like optim)
    def __init__(self, cfg: DictConfig, trainer: pl.Trainer):
        # Check instance type at the very start of init
        self._check_lightning_instance("_init_start")

        # Pass the model-specific config and trainer to ModelPT's init
        super().__init__(cfg=cfg, trainer=trainer)

        # Check instance type after super().__init__
        self._check_lightning_instance("_init_after_super")

    def _check_lightning_instance(self, location: str):
        """Helper to check if the instance is a LightningModule and report."""
        is_lightning = isinstance(self, pl.LightningModule)
        logging.info(
            f"Check in {location}: is instance of pl.LightningModule? {is_lightning}"
        )
        return is_lightning

    # --- ModelPT data setup methods (Required abstract methods) ---
    def setup_training_data(self, train_data_config: Union[DictConfig, Dict, None]):
        # Check instance type at the start of setup_training_data
        self._check_lightning_instance("setup_training_data_start")
        logging.info("setup_training_data called (minimal implementation).")

    def setup_validation_data(self, val_data_config: Union[DictConfig, Dict, None]):
        # Check instance type at the start of setup_validation_data
        self._check_lightning_instance("setup_validation_data_start")
        logging.info("setup_validation_data called (minimal implementation).")

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict, None]):
        # Check instance type at the start of setup_test_data
        self._check_lightning_instance("setup_test_data_start")
        logging.info("setup_test_data called (minimal implementation).")

    # --- Other required abstract methods from ModelPT ---
    @classmethod
    def list_available_models(cls) -> Optional[List[Tuple[str, str]]]:
        return None


# 3. Main execution block to test
if __name__ == "__main__":
    logging.info("Testing MinimalModelPT with PyTorch Lightning Trainer...")

    # Create a minimal experiment config structure
    full_experiment_cfg = OmegaConf.create(
        {
            "model": {  # Model specific config nested under 'model' key
                "optim": {"lr": 0.01},
                # Other model params would go here (e.g., arch)
            },
        }
    )

    # Create a minimal trainer using default arguments
    trainer = pl.Trainer()

    try:
        logging.info("Instantiating MinimalModelPT...")
        # Instantiate with the model-specific config section and the trainer instance.
        model = MinimalModelPT(cfg=full_experiment_cfg.model, trainer=trainer)
        logging.info("MinimalModelPT instantiated successfully.")

        # Check instance type immediately after instantiation
        model._check_lightning_instance("after_instantiation")

        # Manually call setup methods (passing None as data config isn't needed for this minimal test)
        logging.info("Manually calling setup methods...")
        model.setup_training_data(None)
        model.setup_validation_data(None)
        model.setup_test_data(None)
        logging.info("Setup methods called.")

        # Now call trainer.fit. The error we are targeting happens here.
        logging.info("Calling trainer.fit...")
        trainer.fit(model)
        logging.info("trainer.fit completed successfully (should not reach here).")

    except TypeError as e:
        logging.error(f"Caught a TypeError: {e}")
        logging.error("This suggests an issue with type checking.")
        import traceback

        traceback.print_exc()
    except ValueError as e:
        logging.error(f"Caught a ValueError: {e}")
        logging.error("This suggests a configuration structure issue.")
        import traceback

        traceback.print_exc()
    except Exception as e:
        logging.error(f"Caught an unexpected error: {e}")
        import traceback

        traceback.print_exc()

    logging.info("Test finished.")
