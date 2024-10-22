import abc
from typing import Any, Dict, Iterable, Optional

import datasets
from torch.utils.data import IterDataPipe

from oumi.utils.logging import logger


class BaseIterableDataset(IterDataPipe, abc.ABC):
    """Abstract base class for iterable datasets."""

    dataset_name: str
    dataset_path: Optional[str] = None
    default_dataset: Optional[str] = None
    default_subset: Optional[str] = None
    trust_remote_code: bool

    def __init__(
        self,
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
        stream: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseIterableDataset class."""
        dataset_type_name = self.__class__.__name__
        logger.info(f"Creating iterable dataset (type: {dataset_type_name})...")
        if len(kwargs) > 0:
            logger.debug(
                f"Unknown arguments: {', '.join(kwargs.keys())}. "
                "Please check the class constructor for supported arguments "
                f"(type: {dataset_type_name})."
            )

        dataset_name = dataset_name or self.default_dataset

        if dataset_name is None:
            raise ValueError(
                "Please specify a dataset_name or "
                "set the default_dataset class attribute "
                f"(type: {dataset_type_name})."
            )

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_subset = subset or self.default_subset
        self.split = split
        self.stream = stream
        self._data = self._load_data()

    #
    # Main API
    #
    def __iter__(self):
        """Iterates over the dataset."""
        for item in self.data:
            yield self.transform(item)

    def iter_raw(self):
        """Iterates over the raw dataset."""
        yield from self.data

    def to_hf(self) -> datasets.IterableDataset:
        """Converts the dataset to a Hugging Face dataset."""
        return datasets.IterableDataset.from_generator(self.__iter__)

    @property
    def data(self) -> Iterable[Any]:
        """Returns the underlying dataset data."""
        return self._data

    #
    # Abstract Methods
    #
    @abc.abstractmethod
    def transform(self, sample: Any) -> Dict[str, Any]:
        """Preprocesses the inputs in the given sample.

        Args:
            sample (Any): A sample from the dataset.

        Returns:
            dict: A dictionary containing the preprocessed input data.
        """
        raise NotImplementedError

    def _load_data(self) -> Iterable[Any]:
        """Loads the dataset from the specified source."""
        if self.dataset_path:
            result = self._load_local_dataset(self.dataset_path)
        else:
            result = self._load_hf_hub_dataset()

        return result

    def _load_hf_hub_dataset(self) -> Iterable[Any]:
        """Loads the dataset from the specified source."""
        return datasets.load_dataset(
            path=self.dataset_name,
            name=self.dataset_subset,
            split=self.split,
            streaming=self.stream,
        )

    def _load_dataset_from_disk(self, path: str) -> Iterable[Any]:
        return datasets.Dataset.load_from_disk(path)