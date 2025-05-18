# multiblock_train.py
import lightning.pytorch as pl
import numpy as np
import torch
from multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf

@hydra_runner(config_path=".", config_name="multiblock_config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if data needs to be prepared
    import os

    # GE: How is logger setup? What is printed?
    if not os.path.exists("data/sine_train.npz"):
        logging.info("Data files not found. Creating datasets...")
        prepare_data()

    # Set up trainer with Lightning
    # GE:  What are arguments of pl.Trainer?
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model
    # GE: look into trainer in base model
    model = MultiBlockRegressorModel(cfg=cfg, trainer=trainer)

    # Training
    logging.info("Starting training...")
    # GE lightning model
    trainer.fit(model)
    logging.info("Training completed.")

    # Testing
    if hasattr(cfg, "test_ds") and cfg.test_ds.file_path is not None:
        # GE: Should reach this point
        logging.info("Running testing...")
        # GE: Lightning
        trainer.test(model)

    # Save the model if a path is specified
    if (
        hasattr(cfg, "io")
        and hasattr(cfg.io, "nemo_path")
        and cfg.io.nemo_path is not None
    ):
        model.save_to(cfg.io.nemo_path)
        logging.info(f"Model saved to {cfg.io.nemo_path}")
    elif (
        hasattr(cfg, "model")
        and hasattr(cfg.model, "nemo_path")
        and cfg.model.nemo_path is not None
    ):
        model.save_to(cfg.model.nemo_path)
        logging.info(f"Model saved to {cfg.model.nemo_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
