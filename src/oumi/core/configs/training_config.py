from dataclasses import dataclass, field

import torch

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.data_params import DataParams
from oumi.core.configs.params.fsdp_params import FSDPParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.peft_params import PeftParams
from oumi.core.configs.params.training_params import (
    MixedPrecisionDtype,
    TrainerType,
    TrainingParams,
)
from oumi.utils.logging import logger


@dataclass
class TrainingConfig(BaseConfig):
    data: DataParams = field(default_factory=DataParams)
    """Configuration parameters for the dataset.

    This field contains all the necessary settings for data processing and loading.
    It includes options for train and evaluation datasets and preprocessing steps.

    For more details, see the :class:`oumi.core.configs.params.data_params.DataParams`
    class.
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model.

    This field defines the model architecture, size, and other model-specific settings.
    It includes options for model type, pretrained weights, and tokenizer configuration.

    For more details, see :class:`oumi.core.configs.params.model_params.ModelParams`
    class.
    """

    training: TrainingParams = field(default_factory=TrainingParams)
    """Configuration parameters for the training process.

    This field contains all settings related to the training loop,
    including learning rate, batch size, number of epochs, and optimization parameters.

    For more details, see
    :class:`oumi.core.configs.params.training_params.TrainingParams`.
    """

    peft: PeftParams = field(default_factory=PeftParams)
    """Configuration parameters for Parameter-Efficient Fine-Tuning (PEFT).

    This field defines settings for various PEFT methods such as LoRA, or Prefix Tuning.
    It includes options for rank, alpha values, and other PEFT-specific parameters.

    For more details, see :class:`oumi.core.configs.params.peft_params.PeftParams`.
    """

    fsdp: FSDPParams = field(default_factory=FSDPParams)
    """Configuration options for FSDP."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.model.compile:
            raise ValueError(
                "Use `training.compile` instead of `model.compile` to "
                "enable model compilation during training."
            )

        # Verify dataset-related params for TRL_SFT.
        if self.training.trainer_type == TrainerType.TRL_SFT:
            if not self.data.train.target_col:
                raise ValueError("`target_col` must be specified for TRL_SFT Trainer.")

            # Set `dataset_text_field` in `trainer_kwargs` since it's required for
            # `SFTTrainer`, and warn users if their value will be overridden.
            existing_dataset_text_field = self.training.trainer_kwargs.get(
                "dataset_text_field"
            )
            if (existing_dataset_text_field is not None) and (
                existing_dataset_text_field != self.data.train.target_col
            ):
                logger.warning(
                    "Overriding existing `dataset_text_field` value "
                    f"'{existing_dataset_text_field}' with "
                    f"'{self.data.train.target_col}'"
                )
            self.training.trainer_kwargs["dataset_text_field"] = (
                self.data.train.target_col
            )

        # Verify values for model dtype and mixed precision training.
        if self.training.mixed_precision_dtype in [
            MixedPrecisionDtype.FP16,
            MixedPrecisionDtype.BF16,
        ]:
            if self.model.torch_dtype() != torch.float32:
                raise ValueError(
                    "Model must be loaded in fp32 to enable mixed precision training."
                )

        # Check values for model sequence length.
        if self.model.model_max_length and self.model.model_max_length > 0:
            max_seq_length_value = int(self.model.model_max_length)
            max_seq_length_key = None
            if self.training.trainer_type == TrainerType.TRL_SFT:
                max_seq_length_key = "max_seq_length"
            elif self.training.trainer_type == TrainerType.TRL_DPO:
                max_seq_length_key = "max_length"
                # TODO: DPOTrainer also defines "max_prompt_length" and
                # "max_target_length". How to handle them?
            else:
                logger.warning(
                    f"Ignored model.model_max_length={max_seq_length_value} config "
                    f"parameter for trainer {self.training.trainer_type}."
                )

            if max_seq_length_key:
                existing_max_seq_length = self.training.trainer_kwargs.get(
                    max_seq_length_key
                )
                if (existing_max_seq_length is not None) and (
                    existing_max_seq_length != max_seq_length_value
                ):
                    logger.warning(
                        f"Overriding existing '{max_seq_length_key}' value "
                        f"'{existing_max_seq_length}' with '{max_seq_length_value}'"
                    )
                self.training.trainer_kwargs[max_seq_length_key] = max_seq_length_value