from omegaconf import OmegaConf

from gordon.simple_regression_with_adapter.custom_adapter import (
    CustomAdapter,
)
from gordon.simple_regression_with_adapter.parallel_adapter_strategy import (
    ParallelInputAdapterStrategy,
)


# Let's create a utility for building the class path
def get_class_path(cls):
    return f"{cls.__module__}.{cls.__name__}"


common_config = OmegaConf.create(
    {
        "n_layers": 1,
        # I should try ot set this from the command line or from yaml file, leaving
        # it set to 32 in this file
        "hidden_dim": 32,
    }
)

model_config = OmegaConf.create({"model": common_config})

# Input data the conventional way
simple_regressor_config = OmegaConf.create({})

# How to set the ParallelInputAdapterStrategy
custom_adapter_config = OmegaConf.create(
    {
        "_target_": get_class_path(CustomAdapter),
        "size": 1,
        "hidden_dim": "${model.hidden_dim}",
        "first_linear_bias": True,
        "second_linear_bias": True,
        "weight_init_method": "zeros",
        "activation_type": "tanh",
        "adapter_strategy": {
            "_target_": get_class_path(ParallelInputAdapterStrategy),
            "scaling_factor": 1.0,
            "in_features": 2,
            "out_features": 1,
            "bias": True,
        }
    }
)

multiblock_regressor_config = OmegaConf.create(
{
    
}
)


model_config.model.simple_regressor = simple_regressor_config
model_config.model.custom_adapter = custom_adapter_config

# ----------------------------------------------------------------------
