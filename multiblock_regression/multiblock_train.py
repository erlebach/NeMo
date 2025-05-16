# multiblock_train.py
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def create_sine_data(n=1000, noise=0.1):
    """Create sine wave dataset.
    
    Args:
        n: Number of data points
        noise: Noise level
        
    Returns:
        Tuple of inputs and targets
    """
    x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)
    y = np.sin(x) + noise * np.random.randn(*x.shape)
    return x, y


def prepare_data(train_size=1000, val_size=200, test_size=200, noise=0.1):
    """Prepare datasets for training, validation and testing.
    
    Args:
        train_size: Size of training set
        val_size: Size of validation set
        test_size: Size of test set
        noise: Noise level
    """
    # Create directories if they don't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Generate data
    x_train, y_train = create_sine_data(n=train_size, noise=noise)
    x_val, y_val = create_sine_data(n=val_size, noise=noise)
    x_test, y_test = create_sine_data(n=test_size, noise=noise)
    
    # Save datasets
    np.savez("data/sine_train.npz", x=x_train, y=y_train)
    np.savez("data/sine_val.npz", x=x_val, y=y_val)
    np.savez("data/sine_test.npz", x=x_test, y=y_test)
    
    logging.info("Datasets created and saved to the 'data' directory.")


@hydra_runner(config_path=".", config_name="multiblock_config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    logging.info(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    
    # Check if data needs to be prepared
    import os
    if not os.path.exists("data/sine_train.npz"):
        logging.info("Data files not found. Creating datasets...")
        prepare_data()
    
    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Create model
    model = MultiBlockRegressorModel(cfg=cfg, trainer=trainer)
    
    # Training
    logging.info("Starting training...")
    trainer.fit(model)
    logging.info("Training completed.")
    
    # Testing
    if hasattr(cfg, 'test_ds') and cfg.test_ds.file_path is not None:
        logging.info("Running testing...")
        trainer.test(model)
    
    # Save the model if a path is specified
    if hasattr(cfg, 'nemo_path') and cfg.nemo_path is not None:
        model.save_to(cfg.model.nemo_path)
        logging.info(f"Model saved to {cfg.model.nemo_path}")


if __name__ == "__main__":
    main()
