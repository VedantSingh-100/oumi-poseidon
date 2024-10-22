from typing import Optional

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


class BaseExperimentalDpoDataset(BaseMapDataset):
    """Preprocess the samples to the Oumi format.

    Warning:
        This class is experimental and subject to change.
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        return_tensors: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        if return_tensors:
            raise NotImplementedError(
                "return_tensors=True is not implemented for this class"
            )

        self._tokenizer = tokenizer
        self._return_tensors = return_tensors

        self._data = self._load_data()

    def transform_preference(self, samples: dict) -> dict:
        """Transform the samples to the Oumi format."""
        prompt = samples[_PROMPT_KEY]
        chosen_chat = samples[_CHOSEN_KEY]
        rejected_chat = samples[_REJECTED_KEY]

        chosen_chat_response = self._extract_from_chat_format(chosen_chat)
        rejected_chat_response = self._extract_from_chat_format(rejected_chat)

        return {
            _PROMPT_KEY: prompt,
            _CHOSEN_KEY: chosen_chat_response,
            _REJECTED_KEY: rejected_chat_response,
        }

    def transform(self, sample: dict) -> dict:
        """Transform the samples to the Oumi format."""
        return self.transform_preference(sample)

    def _extract_from_chat_format(self, sample: dict) -> str:
        """Extract the last 'assistant' turn in the chat."""
        for turn in sample[::-1]:
            if turn[_ROLE] == _ASSISTANT:
                return turn[_CONTENT]

        raise ValueError("No chat turn was found with an 'assistant' role.")