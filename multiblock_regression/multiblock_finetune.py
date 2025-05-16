# multiblock_finetune.py
import os

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def create_modified_sine_data(n=1000, noise=0.1, phase=0.5):
    """Create modified sine wave dataset with phase shift.
    
    Args:
        n: Number of data points
        noise: Noise level
        phase: Phase shift
        
    Returns:
        Tuple of inputs and targets
    """
    x = np.linspace(-np.pi, np.pi, n).reshape(-1, 1)
    y = np.sin(x + phase) + noise * np.random.randn(*x.shape)
    return x, y


def prepare_finetune_data(train_size=1000, val_size=200, test_size=200, noise=0.1, phase=0.5):
    """Prepare datasets for fine-tuning.
    
    Args:
        train_size: Size of training set
        val_size: Size of validation set
        test_size: Size of test set
        noise: Noise level
        phase: Phase shift
    """
    # Create directories if they don't exist
    os.makedirs("data/finetune", exist_ok=True)
    
    # Generate data
    x_train, y_train = create_modified_sine_data(n=train_size, noise=noise, phase=phase)
    x_val, y_val = create_modified_sine_data(n=val_size, noise=noise, phase=phase)
    x_test, y_test = create_modified_sine_data(n=test_size, noise=noise, phase=phase)
    
    # Save datasets
    np.savez("data/finetune/sine_train.npz", x=x_train, y=y_train)
    np.savez("data/finetune/sine_val.npz", x=x_val, y=y_val)
    np.savez("data/finetune/sine_test.npz", x=x_test, y=y_test)
    
    logging.info("Fine-tuning datasets created and saved to the 'data/finetune' directory.")


@hydra_runner(config_path="conf", config_name="multiblock_finetune_config")
def main(cfg: DictConfig) -> None:
    """Main fine-tuning function.
    
    Args:
        cfg: Hydra configuration
    """
    logging.info(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    
    # Check if fine-tuning data needs to be prepared
    if not os.path.exists("data/finetune/sine_train.npz"):
        logging.info("Fine-tuning data files not found. Creating datasets...")
        prepare_finetune_data()
    
    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Create model from checkpoint
    model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.restore_path,
        trainer=trainer
    )
    
    # Add and enable the adapter
    if hasattr(cfg, 'adapter'):
        adapter_cfg = cfg.adapter
        model.add_adapter(adapter_cfg.name, adapter_cfg)
        model.set_enabled_adapters(adapter_cfg.name, enabled=True)
        
        # Freeze base model parameters
        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
    
    # Training
    logging.info("Starting fine-tuning...")
    trainer.fit(model)
    logging.info("Fine-tuning completed.")
    
    # Testing
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.file_path is not None:
        logging.info("Running testing...")
        trainer.test(model)
    
    # Save the model if a path is specified
    if cfg.model.get('nemo_path'):
        model.save_to(cfg.model.nemo_path)
        logging.info(f"Adapter model saved to {cfg.model.nemo_path}")


if __name__ == "__main__":
    main()
