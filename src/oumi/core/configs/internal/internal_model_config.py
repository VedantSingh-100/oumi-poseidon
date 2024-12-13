from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple, Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.constants import LABEL_IGNORE_INDEX


class InternalFeatureFirstDimAction(Enum):
    """Enum representing how to handle the first feature dimension in datasets."""

    DROP_ALWAYS = "drop_always"
    """The first dimension is commonly dummy (length: 1) and must be dropped.

    In effect, this operation is applied: `x = x[0, ...]`, which reduces
    `x`'s rank by 1 (e.g., 3D->2D), and discards the following elements: `x[1:, ...]`.
    """

    DROP_IF_DUMMY = "drop_if_dummy"
    """Drop the first dimension only if it's dummy (length: 1)."""

    KEEP = "keep"
    """Always preserve the first dimension."""


class InternalFeatureSpec(NamedTuple):
    name: str
    """Feature name."""

    required: bool = False
    """Whether the feature must be always present (vs optional)."""

    variable_shape: bool = False
    """Whether the feature can be of variable shape.

    For example, `input_ids` is normally of variable length.
    """

    first_dim_action: InternalFeatureFirstDimAction = (
        InternalFeatureFirstDimAction.DROP_ALWAYS
    )
    """Action to apply to the first feature dimension."""


@dataclass
class InternalVisualModelConfig(BaseConfig):
    variable_shape_image_features: bool = False
    """Whether image features can be of variable shape.

    In this case, the image features can be difficult to collate
    (`torch.stack()` requires compatible shapes) and some workaround
    is needed: either require `batch_size=1`, or group examples
    so that each mini-batch only contains same-sized features.
    """

    supports_multiple_images: bool = False
    """Whether the visual language model supports multiple images in one prompt."""


def _default_model_input_features_factory() -> dict[str, InternalFeatureSpec]:
    result_list: list[InternalFeatureSpec] = [
        InternalFeatureSpec(name="input_ids", required=True, variable_shape=True),
        InternalFeatureSpec(name="attention_mask", required=False, variable_shape=True),
        InternalFeatureSpec(name="labels", required=False, variable_shape=True),
    ]
    return {x.name: x for x in result_list}


@dataclass
class InternalModelConfig(BaseConfig):
    model_type: str = ""
    """Model type."""

    chat_template: str = "default"
    """Default chat template."""

    model_input_features: dict[str, InternalFeatureSpec] = field(
        default_factory=_default_model_input_features_factory
    )
    """Model input features specs."""

    label_ignore_index: Optional[int] = LABEL_IGNORE_INDEX
    """Special label value to be excluded from loss computation."""

    sanitize_negative_labels: bool = False
    """Replace negative label values.

    Some VLM processors can generate negative `input_ids` for image tokens,
    which can cause problems if a negative integer is used as a label
    to compute loss e.g., cross-entropy loss may expect [0, num_classes) range.
    """

    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra params to pass to processor constructor."""

    visual_config: Optional[InternalVisualModelConfig] = None
    """Configuration specific to visual models."""