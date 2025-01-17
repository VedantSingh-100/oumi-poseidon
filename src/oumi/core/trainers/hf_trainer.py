from typing import Optional

import transformers

from oumi.core.configs import TrainingConfig
from oumi.core.distributed import is_world_process_zero
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


class HuggingFaceTrainer(BaseTrainer):
    def __init__(
        self,
        hf_trainer: transformers.Trainer,
        processor: Optional[BaseProcessor] = None,
    ):
        """Initializes HuggingFace-specific Trainer version."""
        self._hf_trainer = hf_trainer
        self._processor = processor

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Trains a model."""
        logger.info(
            f"Starting training with transformers=={transformers.__version__}..."
        )
        self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_state(self) -> None:
        """See base class.

        Saves the Trainer state, since Trainer.save_model saves only the tokenizer
        with the model.

        HuggingFace normally writes state into "trainer_state.json" under output_dir.
        """
        if not is_world_process_zero():
            return

        self._hf_trainer.save_state()

    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's weights to the specified output directory.

        Args:
            config: The Oumi training config.
            final: Whether this is the final model being saved during training.
                - Applies optimizations for the final model checkpoint.
                - In the case of FSDP, this will always save the FULL_STATE_DICT
                instead of the default STATE_DICT.

        Returns:
            None
        """
        if self._hf_trainer.is_fsdp_enabled:
            # FSDP is enabled, so we need to save the model in a special way.
            return self._save_fsdp_model(config=config, final=final)

        if is_world_process_zero():
            output_dir = config.training.output_dir
            self._hf_trainer.save_model(output_dir)
            logger.info(f"Model has been saved at {output_dir}.")

            if self._processor is not None:
                self._processor.save_config(output_dir)
                logger.info(f"Processor config has been saved at {output_dir}.")

    def _save_fsdp_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's weights to the specified output directory.

        For FSDP, all ranks should call into this function
        """
        if final:
            # For the final checkpoint, we need to save the FULL_STATE_DICT instead of
            # the default STATE_DICT.
            if (
                self._hf_trainer.is_fsdp_enabled
                and self._hf_trainer.accelerator.state.fsdp_plugin is not None
            ):
                logger.info("Saving FULL_STATE_DICT for final model checkpoint.")
                self._hf_trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
                    "FULL_STATE_DICT"
                )

        output_dir = config.training.output_dir
        self._hf_trainer.save_model(output_dir)
        logger.info(f"Model has been saved at {output_dir}.")
