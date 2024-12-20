import sys
sys.path.append('/home/vhsingh/oumi/src')  # Add 'src' to Python's path

from oumi.core.registry import register, RegistryType
from oumi.models.Poseidon.scOT.model import ScOT, ScOTConfig
import torch

@register(registry_name="poseidon_pde", registry_type=RegistryType.MODEL)
def build_poseidon_model(model_params, peft_params=None, **kwargs):
    """
    Registers and builds the Poseidon (ScOT) model within the OUMI framework.
    """

    # Extract configuration parameters
    config_kwargs = model_params.model_kwargs
    pretrained_weights = model_params.load_pretrained_weights
    model_name = model_params.model_name

    # Handle PEFT/adapter model flags as per OUMI’s logic
    if peft_params and peft_params.q_lora:
        raise NotImplementedError("Q-LoRA is not supported for Poseidon model yet.")

    if model_params.adapter_model is not None:
        raise NotImplementedError("Adapter models are not supported for Poseidon model yet.")

    # Build the configuration
    config = ScOTConfig(**config_kwargs)

    # Load model
    if pretrained_weights:
        # Ensure that from_pretrained is implemented and model_name points to a valid source
        model = ScOT.from_pretrained(model_name, config=config)
    else:
        model = ScOT(config)

    # Return the raw model; OUMI will handle dtype and other configurations afterwards.
    return model
