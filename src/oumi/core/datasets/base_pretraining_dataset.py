from typing import Any, Dict, List

import torch
from typing_extensions import override

from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.tokenizers import BaseTokenizer


class BasePretrainingDataset(BaseIterableDataset):
    """Abstract base class for pretraining iterable datasets.

    This class extends BaseIterableDataset to provide functionality specific to
    pretraining tasks.

    Attributes:
        tokenizer (BaseTokenizer): The tokenizer used for text encoding.
        seq_length (int): The desired sequence length for model inputs.
        concat_token_id (int): The ID of the token used to concatenate documents.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from oumi.core.datasets import PretrainingDataset
        >>>
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> dataset = PretrainingDataset(
        ...     dataset_name="wikimedia/wikipedia",
        ...     subset="20231101.en",
        ...     tokenizer=tokenizer,
        ...     seq_length=512
        ... )
        >>>
        >>> for batch in dataset:
        ...     print(batch)  # Process the batch
    """

    def __init__(
        self,
        *,
        tokenizer: BaseTokenizer,
        seq_length: int,
        dataset_text_field: str = "text",
        append_concat_token: bool = True,
        add_special_tokens: bool = True,
        skip_last: bool = True,
        **kwargs,
    ):
        """Initializes a new instance of the BasePretrainingDataset class."""
        if append_concat_token and tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must have an EOS token if append_concat_token is enabled."
            )

        self.concat_token_id = tokenizer.eos_token_id if append_concat_token else None

        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self._dataset_text_field = dataset_text_field
        self._append_concat_token = append_concat_token
        self._add_special_tokens = add_special_tokens
        self._skip_last = skip_last

        super().__init__(**kwargs)

    def __iter__(self):
        """Iterates over the dataset and yields samples of a specified sequence length.

        The underlying dataset is a stream of documents. Each document is expected to
        containt a text field `self._dataset_text_field` that will be tokenized.
        Training samples are then yielded in sequences of length `self.seq_length`.

        Given this iterator might return samples from different documents, we optionally
        use `self.concat_token_id` to separate the sequences from different documents.
        """
        buffer = []
        for document in self.data:
            if self._append_concat_token and len(buffer) > 0:
                # We started preprocessing a new document
                # so we need to append the concatenation token to mark the end
                # of the previous document.
                buffer.append(self.concat_token_id)

            # Pre-process and tokenize the document
            document_tokens = self.transform(document[self._dataset_text_field])
            buffer.extend(document_tokens)

            # Yield sequences of the specified length.
            # Otherwise, resume pre-processing the next document.
            while len(buffer) >= self.seq_length:
                # We have enough tokens to create a fully packed sample
                yield self._create_training_sample(buffer[: self.seq_length])
                buffer = buffer[self.seq_length :]

        # Finished iterating on the dataset, yield the remaining buffer
        if len(buffer) > 0:
            if not self._skip_last or len(buffer) == self.seq_length:
                yield self._create_training_sample(buffer)

    @override
    def transform(self, sample: Any) -> List[int]:
        """Preprocesses the inputs in the given sample."""
        return self.tokenize(sample)

    def tokenize(self, text: str) -> List[int]:
        """Tokenizes the given text.

        Should not apply any padding/truncation to allow for packing.
        """
        return self.tokenizer.encode(
            text=text,
            return_tensors=None,
            max_length=None,
            padding=False,
            truncation=False,
            add_special_tokens=self._add_special_tokens,
        )

    def _create_training_sample(self, tokens: list) -> Dict[str, torch.Tensor]:
        """Creates a training sample from the given tokens."""
        input_ids = torch.tensor(tokens)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }