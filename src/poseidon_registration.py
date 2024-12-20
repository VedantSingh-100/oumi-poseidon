from oumi.core.registry.registry import register, RegistryType  # Adjusted import path
from oumi.models.Poseidon.scOT.model import ScOT, ScOTConfig  # Adjusted import path
import torch

print("Entered the registry file")

@register(registry_name="poseidon_pde", registry_type=RegistryType.MODEL)
def build_poseidon_model(model_params, peft_params=None, **kwargs):
    """
    Registers and builds the Poseidon (ScOT) model within the OUMI framework.
    """
    print("Registered the file")
    config_kwargs = model_params.model_kwargs or {}
    pretrained_weights = model_params.load_pretrained_weights

    # Create model configuration
    config = ScOTConfig(**config_kwargs)

    # Load model with or without pretrained weights
    if pretrained_weights:
        model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
    else:
        model = ScOT(config)

    return model
