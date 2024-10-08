from typing import Dict, Union, cast

import numpy as np
import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role


@register_dataset("argilla/databricks-dolly-15k-curated-en")
class ArgillaDollyDataset(BaseLMSftDataset):
    """Dataset class for the Databricks Dolly 15k curated dataset."""

    default_dataset = "argilla/databricks-dolly-15k-curated-en"

    def __init__(self, *, use_new_fields: bool = True, **kwargs) -> None:
        """Initialize the DollyDataset.

        Args:
            use_new_fields (bool): Whether to use the new fields
            (new-instruction, new-context, new-response) instead of the original fields.
            Defaults to True.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        self.use_new_fields = use_new_fields
        super().__init__(**kwargs)

    @override
    def transform_conversation(self, example: Union[Dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object.

        Args:
            example: A single example from the dataset.

        Returns:
            Conversation: A Conversation object containing the transformed messages.
        """
        messages = []

        col_prefix = "new" if self.use_new_fields else "original"

        instruction = self._get_field_value(example, f"{col_prefix}-instruction")
        context = self._get_field_value(example, f"{col_prefix}-context")
        response = self._get_field_value(example, f"{col_prefix}-response")

        # Construct the user message
        user_content = instruction
        if context:
            user_content += f"\n\nContext: {context}"

        messages.append(Message(role=Role.USER, content=user_content))
        messages.append(Message(role=Role.ASSISTANT, content=response))

        return Conversation(messages=messages)

    @staticmethod
    def _get_field_value(example: Union[Dict, pd.Series], field: str) -> str:
        """Helper method to get the value from a field.

        Args:
            example (Union[Dict, pd.Series]): A single example from the dataset.
            field (str): The field name to retrieve.

        Returns:
            str: The value of the field.
        """
        value = example[field]

        if isinstance(value, str):
            return value
        if isinstance(value, (dict, pd.Series)) and "value" in value:
            return cast(
                str,
                value["value"][0]
                if isinstance(value["value"], (list, np.ndarray))
                else value["value"],
            )
        raise RuntimeError(f"Unable to parse field: {field}")