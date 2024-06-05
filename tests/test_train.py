import tempfile

from lema import train
from lema.core.types import DataParams, ModelParams, TrainingConfig, TrainingParams


def test_basic_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
            trainer_kwargs={
                "dataset_text_field": "prompt",
            },
        ),
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        training=TrainingParams(
            max_steps=3,
            logging_steps=3,
            enable_wandb=False,
            enable_tensorboard=False,
            output_dir=output_temp_dir,
        ),
    )

    train(config)


def test_custom_train():
    output_temp_dir = tempfile.mkdtemp()

    config: TrainingConfig = TrainingConfig(
        data=DataParams(
            dataset_name="yahma/alpaca-cleaned",
            preprocessing_function_name="alpaca",
            trainer_kwargs={
                "dataset_text_field": "prompt",
            },
        ),
        model=ModelParams(
            model_name="learning-machines/sample",
            tokenizer_name="gpt2",
            trust_remote_code=False,
        ),
        training=TrainingParams(
            max_steps=3,
            logging_steps=3,
            enable_wandb=False,
            enable_tensorboard=False,
            output_dir=output_temp_dir,
            include_performance_metrics=True,
        ),
    )

    train(config)