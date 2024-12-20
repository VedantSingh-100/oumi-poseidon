import sys
sys.path.append('/home/vhsingh/oumi/src')

from oumi.core.configs import TrainingConfig
from oumi.train import train
from oumi.utils.torch_utils import device_cleanup
import poseidon_registration  # This ensures poseidon_pde is registered
import Poseidon_CE_dataset

def main():
    device_cleanup()
    print("Entering the training config")

    # Create a minimal inline training config
    train_config: TrainingConfig = TrainingConfig.from_yaml(
        "/home/vhsingh/oumi/configs/recipes/vision/Poseidon/config.yaml"
    )

    train_config.finalize_and_validate()
    train(train_config)

if __name__ == "__main__":
    main()
