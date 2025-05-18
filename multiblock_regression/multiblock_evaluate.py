# multiblock_evaluate.py
import json
import os
import tarfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from multiblock_regressor import (
    MLP,
    MultiBlockRegressor,  # Import the base regressor directly
    ResidualMLP,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf


@hydra_runner(config_path=".", config_name="multiblock_evaluate_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    try:
        # Load the actual base and adapted models
        logging.info(f"Loading base model from {cfg.model.base_model_path}")
        base_model = MultiBlockRegressor.restore_from(cfg.model.base_model_path)
        base_model.eval()

        logging.info(f"Loading adapted model from {cfg.model.adapter_model_path}")
        adapter_model = MultiBlockRegressor.restore_from(cfg.model.adapter_model_path)
        adapter_model.eval()

        # Generate evaluation points for plotting smooth curves
        x_plot = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
        x_tensor = torch.tensor(x_plot, dtype=torch.float32)

        # Load actual original and modified noisy data for targets
        logging.info(f"Loading original data from {cfg.evaluation.original_data_path}")
        original_data = np.load(cfg.evaluation.original_data_path)
        # We will plot against x_original_data, but run inference on x_plot
        x_original_data = original_data["x"]
        y_original_target = original_data["y"]  # Use this as target for plotting

        logging.info(f"Loading finetune data from {cfg.evaluation.finetune_data_path}")
        finetune_data = np.load(cfg.evaluation.finetune_data_path)
        # We will plot against x_finetune_data, but run inference on x_plot
        x_finetune_data = finetune_data["x"]
        y_modified_target = finetune_data["y"]  # Use this as target for plotting

        with torch.no_grad():
            # Get actual predictions from loaded models using the smooth x_plot points
            y_base_pred = base_model(x_tensor).cpu().numpy()
            y_adapter_pred = adapter_model(x_tensor).cpu().numpy()

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot for Original Sine Task
        # Plot the actual noisy target data points
        axes[0].plot(
            x_original_data,
            y_original_target,
            "bo",
            markersize=3,
            alpha=0.5,
            label="Original Data (Target)",
        )
        axes[0].plot(x_plot, y_base_pred, "r-", label="Base Model Prediction")
        axes[0].plot(x_plot, y_adapter_pred, "m-", label="Adapter Model Prediction")
        axes[0].set_title("Evaluation on Original Sine Task")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot for Modified Sine Task
        # Plot the actual noisy target data points
        axes[1].plot(
            x_finetune_data,
            y_modified_target,
            "go",
            markersize=3,
            alpha=0.5,
            label=f"Modified Data (Phase={cfg.evaluation.phase_shift}, Target)",
        )
        axes[1].plot(x_plot, y_base_pred, "r-", label="Base Model Prediction")
        axes[1].plot(x_plot, y_adapter_pred, "m-", label="Adapter Model Prediction")
        axes[1].set_title(
            f"Evaluation on Modified Sine Task (Phase={cfg.evaluation.phase_shift})"
        )
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Save the figure
        plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels
        plt.savefig(cfg.evaluation.output_figure)
        logging.info(f"Evaluation plot saved to {cfg.evaluation.output_figure}")

    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
