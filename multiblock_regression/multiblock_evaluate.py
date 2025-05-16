# multiblock_evaluate.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from multiblock_adapter import MultiBlockRegressorModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path=".", config_name="multiblock_evaluate_config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function.
    
    Args:
        cfg: Hydra configuration
    """
    logging.info(f'Config:\n{OmegaConf.to_yaml(cfg)}')
    
    # Load base model
    base_model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.base_model_path
    )
    base_model.eval()
    
    # Load adapter model
    adapter_model = MultiBlockRegressorModel.restore_from(
        restore_path=cfg.model.adapter_model_path
    )
    adapter_model.eval()
    
    # Make sure the adapter is enabled
    adapter_names = adapter_model.get_enabled_adapters()
    if len(adapter_names) > 0:
        logging.info(f"Enabled adapters: {adapter_names}")
    else:
        logging.warning("No adapters are enabled in the adapter model.")
    
    # Generate evaluation points
    x = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Original sine function (target for base model)
    y_original = np.sin(x)
    
    # Modified sine function (target for adapter model)
    y_modified = np.sin(x + cfg.evaluation.phase_shift)
    
    # Get model predictions
    with torch.no_grad():
        y_base = base_model(x_tensor).numpy()
        y_adapter = adapter_model(x_tensor).numpy()
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_original, 'b-', label='Original Sine (Base Target)')
    plt.plot(x, y_modified, 'g--', label=f'Modified Sine (Phase={cfg.evaluation.phase_shift}, Adapter Target)')
    plt.plot(x, y_base, 'r-', label='Base Model Prediction')
    plt.plot(x, y_adapter, 'm-', label='Adapter Model Prediction')
    
    # Add legend and labels
    plt.legend(fontsize=12)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Comparison of Base Model and LoRA-adapted Model', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(cfg.evaluation.output_figure)
    logging.info(f"Evaluation plot saved to {cfg.evaluation.output_figure}")


if __name__ == "__main__":
    main()
