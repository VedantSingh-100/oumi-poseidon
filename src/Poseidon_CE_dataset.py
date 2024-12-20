import io
import numpy as np
from typing_extensions import override
from datasets import load_dataset
from transformers import AutoImageProcessor

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    Role,
    Type,
)
@register_dataset("pde_ce_rp")
class PDE_CERP_Dataset(VisionLanguageSftDataset):
    # This is the Hugging Face dataset name as found on the hub
    default_dataset = "camlab-ethz/CE-RP"
    # A default prompt for the user (artificial, just to fit OUMI structure)
    default_prompt = "Simulate the PDE given these initial conditions."

    def __init__(self, split="train", model_name=None,**kwargs):
        """
        Initialize the dataset by loading it from Hugging Face.
        The dataset should have a 'data' field that returns a NumPy array
        or something convertible to NumPy.

        Args:
            split (str): "train", "val", or "test"
            **kwargs: Additional arguments passed to VisionLanguageSftDataset or HF load.
        """
        super().__init__(**kwargs)  # Initialize parent class if needed

        # Load the dataset from Hugging Face
        # Adjust the split mapping if needed
        hf_splits = {
            "train": "train",  # You may need to confirm the actual split names
            "val": "validation",
            "test": "test"
        }
        if split not in hf_splits:
            raise ValueError(f"Unknown split: {split}")

        self.dataset = load_dataset(self.default_dataset, split=hf_splits[split])

        # Integrate AutoImageProcessor for SwinV2
        if model_name:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        else:
            self.image_processor = None

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """
        Transform a single example into a Conversation object.
        
        This version integrates AutoImageProcessor for SwinV2 preprocessing.
        """
        if "data" not in example:
            raise ValueError(
                "Expected 'data' key in example. "
                f"Available keys: {example.keys()}."
            )

        trajectory = example["data"]  # [21, 5, 128, 128] (example)
        if trajectory.shape != (21, 5, 128, 128):
            raise ValueError(
                f"Expected trajectory shape (21,5,128,128), got {trajectory.shape}."
            )

        # Separate initial conditions and the rest of the evolution
        initial_conditions = trajectory[0]  # [5, 128, 128]
        evolution = trajectory[1:]          # [20, 5, 128, 128]

        # Preprocess using AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

        # Assuming SwinV2 can process single channels or stacked channels
        # Normalize and resize spatial dimensions as needed (example)
        initial_conditions_processed = image_processor(
            images=[initial_conditions], return_tensors="pt"
        )
        evolution_processed = image_processor(
            images=[evolution], return_tensors="pt"
        )

        # Convert processed tensors back into binary form if needed
        initial_bytes = io.BytesIO()
        np.save(initial_bytes, initial_conditions_processed['pixel_values'].numpy())
        initial_content = initial_bytes.getvalue()

        evolution_bytes = io.BytesIO()
        np.save(evolution_bytes, evolution_processed['pixel_values'].numpy())
        evolution_content = evolution_bytes.getvalue()

        # Create user message
        user_items = [
            MessageContentItem(
                type=Type.TEXT,
                content=self.default_prompt
            ),
            MessageContentItem(
                type=Type.IMAGE_BINARY,
                binary=initial_content
            )
        ]

        # Create assistant message
        assistant_items = [
            MessageContentItem(
                type=Type.IMAGE_BINARY,
                binary=evolution_content
            )
        ]

        return Conversation(
            messages=[
                Message(role=Role.USER, type=Type.COMPOUND, content=user_items),
                Message(role=Role.ASSISTANT, type=Type.COMPOUND, content=assistant_items),
            ]
        )
