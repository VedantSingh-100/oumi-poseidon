from abc import ABC, abstractmethod
from typing import Optional

from oumi.core.configs import TrainingConfig


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, resume_from_checkpoint: Optional[str]) -> None:
        """Trains a model."""

    @abstractmethod
    def save_state(self) -> None:
        """Saves the Trainer state.

        Under distributed environment this is done only for a process with rank 0.
        """
        # TODO: Define semantics of this method more clearly.
        # Can it be merged with save_model()?

    @abstractmethod
    def save_model(self, config: TrainingConfig, final: bool = True) -> None:
        """Saves the model's state dictionary to the specified output directory.

        Args:
            config (TrainingConfig): The Oumi training config.
            final (bool): Whether this is the final model being saved during training.

        Returns:
            None
        """
        # TODO: Define semantics of this method more clearly.
