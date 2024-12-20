from oumi.core.configs import EvaluationConfig, InferenceConfig, TrainingConfig
from oumi.evaluate import evaluate
from oumi.infer import infer
from oumi.train import train
from oumi.utils.torch_utils import device_cleanup
import poseidon_registration

def main() -> None:
    """Run Llama 1B train/eval/infer."""
    model_output_dir = "output/Poseidon"
    device_cleanup()
    train_config: TrainingConfig = TrainingConfig.from_yaml(
        "/home/vhsingh/oumi/configs/recipes/vision/Poseidon/config.yaml"
    )
    train_config.training.enable_wandb = False
    train_config.training.max_steps = 100
    train_config.training.output_dir = model_output_dir
    train_config.finalize_and_validate()
    train(train_config)

if __name__ == "__main__":
    main()