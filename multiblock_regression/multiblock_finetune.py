# multiblock_finetune.py
import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from multiblock_regression.multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from omegaconf import DictConfig, OmegaConf


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


def prepare_finetune_data(
    train_size=1000, val_size=200, test_size=200, noise=0.1, phase=0.5
):
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
    # Same sine wave for all three, different noise
    x_train, y_train = create_modified_sine_data(n=train_size, noise=noise, phase=phase)
    x_val, y_val = create_modified_sine_data(n=val_size, noise=noise, phase=phase)
    x_test, y_test = create_modified_sine_data(n=test_size, noise=noise, phase=phase)

    # Save datasets
    np.savez("data/finetune/sine_train.npz", x=x_train, y=y_train)
    np.savez("data/finetune/sine_val.npz", x=x_val, y=y_val)
    np.savez("data/finetune/sine_test.npz", x=x_test, y=y_test)

    logging.info(
        "Fine-tuning datasets created and saved to the 'data/finetune' directory."
    )


@hydra_runner(config_path=".", config_name="multiblock_finetune_config")
def main(cfg: DictConfig) -> None:
    """Main fine-tuning function.

    Args:
        cfg: Hydra configuration
    """
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if fine-tuning data needs to be prepared
    # if not os.path.exists("data/finetune/sine_train.npz"):
    if not Path("data/finetune/sine_train.npz").exists():
        logging.info("Fine-tuning data files not found. Creating datasets...")
        prepare_finetune_data()

    # Set up trainer with Lightning
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Create model from checkpoint
    model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.restore_path, trainer=trainer
    )

    # Convert config to container and back to make it mutable
    model_cfg = OmegaConf.create(OmegaConf.to_container(model.cfg, resolve=True))

    # Now create the model namespace if needed
    if not hasattr(model_cfg, "model"):
        model_cfg.model = {}

    # Manually copy each dataloader config to avoid structure errors
    if hasattr(cfg, "model"):
        if hasattr(cfg.model, "train_ds"):
            train_ds_dict = OmegaConf.to_container(cfg.model.train_ds, resolve=True)
            model_cfg.model.train_ds = OmegaConf.create(train_ds_dict)

        if hasattr(cfg.model, "validation_ds"):
            val_ds_dict = OmegaConf.to_container(cfg.model.validation_ds, resolve=True)
            model_cfg.model.validation_ds = OmegaConf.create(val_ds_dict)

        if hasattr(cfg.model, "test_ds"):
            test_ds_dict = OmegaConf.to_container(cfg.model.test_ds, resolve=True)
            model_cfg.model.test_ds = OmegaConf.create(test_ds_dict)

    # Assign the updated config back to the model
    model.cfg = model_cfg

    # ==== DIRECT DATALOADER CREATION ====
    import numpy as np
    import torch

    # Manually create dataloaders directly
    try:
        print(f"Trying to load training data from: {cfg.model.train_ds.file_path}")
        train_data = np.load(cfg.model.train_ds.file_path)
        train_x = train_data["x"]
        train_y = train_data["y"]
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        )
        model._train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.model.train_ds.batch_size,
            shuffle=True,
            num_workers=cfg.model.train_ds.get("num_workers", 0),
            pin_memory=cfg.model.train_ds.get("pin_memory", False),
        )
        print("Training dataloader created successfully.")
    except Exception as e:
        print(f"Error creating training dataloader: {e}")

    try:
        print(
            f"Trying to load validation data from: {cfg.model.validation_ds.file_path}"
        )
        val_data = np.load(cfg.model.validation_ds.file_path)
        val_x = val_data["x"]
        val_y = val_data["y"]
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=torch.float32),
        )
        model._validation_dl = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.model.validation_ds.batch_size,
            shuffle=False,
            num_workers=cfg.model.validation_ds.get("num_workers", 0),
            pin_memory=cfg.model.validation_ds.get("pin_memory", False),
        )
        print("Validation dataloader created successfully.")
    except Exception as e:
        print(f"Error creating validation dataloader: {e}")

    # Force override the val_dataloader method temporarily
    def fixed_val_dataloader(self):
        print("Using fixed val_dataloader method")
        return self._validation_dl

    import types

    model.val_dataloader = types.MethodType(fixed_val_dataloader, model)

    # Try to verify dataloaders are created
    print(f"Training dataloader exists: {model._train_dl is not None}")
    print(f"Validation dataloader exists: {model._validation_dl is not None}")

    # Add and enable the adapter
    if hasattr(cfg, "adapter"):
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
    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.file_path is not None:
        logging.info("Running testing...")
        trainer.test(model)

    # Save the model if a path is specified
    if cfg.model.get("nemo_path"):
        model.save_to(cfg.model.nemo_path)
        logging.info(f"Adapter model saved to {cfg.model.nemo_path}")

    # Load base model
    base_model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.base_model_path
    )
    # Make sure required architecture parameters exist
    if not hasattr(base_model.cfg, "arch"):
        base_model.cfg.arch = OmegaConf.create(
            {
                "input_dim": 1,
                "hidden_dim": 16,
                "output_dim": 1,
                "num_blocks": 2,
                "num_layers_per_block": 2,
                "activation": "tanh",
            }
        )
    base_model.eval()

    # Load adapter model (same approach)
    adapter_model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.adapter_model_path
    )
    # Make sure required architecture parameters exist
    if not hasattr(adapter_model.cfg, "arch"):
        adapter_model.cfg.arch = OmegaConf.create(
            {
                "input_dim": 1,
                "hidden_dim": 16,
                "output_dim": 1,
                "num_blocks": 2,
                "num_layers_per_block": 2,
                "activation": "tanh",
            }
        )
    adapter_model.eval()


if __name__ == "__main__":
    main()
