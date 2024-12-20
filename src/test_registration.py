import sys
sys.path.append('/home/vhsingh/oumi/src')  # Add 'src' to Python's path

from oumi.core.registry.registry import REGISTRY, RegistryType
from oumi.models.Poseidon.scOT.model import ScOT, ScOTConfig  # Ensure correct imports
import poseidon_registration  # Import the file that registers the model

def test_registration():
    """
    Test if the custom Poseidon model is properly registered in the Oumi registry.
    """
    # Registry key to check
    model_name = "pde_ce_rp"
    registry_type = RegistryType.DATASET

    # Check if the model is registered
    if REGISTRY.contains(model_name, registry_type):
        print(f"SUCCESS: `{model_name}` is registered under `{registry_type.name}`!")
    else:
        print(f"FAILURE: `{model_name}` is NOT registered under `{registry_type.name}`.")

    # Optionally, list all models in the registry
    all_models = REGISTRY.get_all(RegistryType.MODEL)
    print("Currently registered models:")
    for name, model in all_models.items():
        print(f" - {name}: {model}")

if __name__ == "__main__":
    test_registration()
