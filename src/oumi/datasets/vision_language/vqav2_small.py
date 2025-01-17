from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    Role,
    Type,
)


@register_dataset("merve/vqav2-small")
class Vqav2SmallDataset(VisionLanguageSftDataset):
    default_dataset = "merve/vqav2-small"

    def _process_text_value(self, s: str) -> str:
        # The data contains occasional `\n` at the beginning or end
        # of text values. Let's strip them.
        return s.strip() if s else ""

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self._process_text_value(example["question"])
        output_text = self._process_text_value(example["multiple_choice_answer"])

        messages = [
            Message(
                role=Role.USER,
                type=Type.COMPOUND,
                content=[
                    MessageContentItem(
                        type=Type.IMAGE_BINARY,
                        binary=example["image"]["bytes"],
                    ),
                    MessageContentItem(type=Type.TEXT, content=input_text),
                ],
            ),
            Message(role=Role.ASSISTANT, type=Type.TEXT, content=output_text),
        ]

        return Conversation(messages=messages)
