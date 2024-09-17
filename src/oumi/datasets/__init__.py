"""Datasets module for the OUMI (Open Unified Machine Intelligence) library.

This module provides various dataset implementations for use in the OUMI framework.
These datasets are designed for different machine learning tasks and can be used
with the models and training pipelines provided by OUMI.

For more information on the available datasets and their usage, see the
:mod:`oumi.datasets` documentation.

Each dataset is implemented as a separate class, inheriting from appropriate base
classes in the :mod:`oumi.core.datasets` module. For usage examples and more detailed
information on each dataset, please refer to their respective class documentation.

See Also:
    - :mod:`oumi.models`: Compatible models for use with these datasets.
    - :mod:`oumi.core.datasets`: Base classes for dataset implementations.

Example:
    >>> from oumi.datasets import AlpacaDataset
    >>> dataset = AlpacaDataset()
    >>> train_loader = DataLoader(dataset, batch_size=32)
"""

from oumi.datasets.alpaca import AlpacaDataset
from oumi.datasets.chatqa import ChatqaDataset
from oumi.datasets.chatrag_bench import ChatRAGBenchDataset
from oumi.datasets.debug import DebugClassificationDataset, DebugPretrainingDataset
from oumi.datasets.vision_language.coco_captions import COCOCaptionsDataset
from oumi.datasets.vision_language.flickr30k import Flickr30kDataset
from oumi.datasets.vision_language.vision_jsonlines import JsonlinesDataset

__all__ = [
    "AlpacaDataset",
    "ChatqaDataset",
    "ChatRAGBenchDataset",
    "DebugClassificationDataset",
    "DebugPretrainingDataset",
    "COCOCaptionsDataset",
    "Flickr30kDataset",
    "JsonlinesDataset",
]