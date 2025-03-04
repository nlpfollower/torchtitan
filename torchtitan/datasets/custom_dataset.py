from typing import List, Dict, Any, Optional

import torch
from click.core import batch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import broadcast
from torch.utils.data import DataLoader, IterableDataset
from torchdata.nodes import Stateful

from torchtitan.datasets.hh_dataset import build_hh_data_loader, HHDataset
from torchtitan.datasets.mmlu_dataset import MMLUDataset
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import logger

class CustomIterableDataset(IterableDataset, Stateful):
    def __init__(
            self,
            dataset,
            seq_len: int = 2048,
            world_size: int = 1,
            rank: int = 0,
            infinite: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.infinite = infinite
        self.world_size = world_size
        self.rank = rank
        self._sample_idx = 0

    def __iter__(self):
        while True:
            for i, sample in enumerate(self.dataset[self._sample_idx:]):
                if i % self.world_size == self.rank:
                    self._sample_idx += 1
                    yield sample['input_ids'], sample['labels']

            if not self.infinite:
                logger.warning("CustomIterableDataset has exhausted its data.")
                break
            else:
                logger.warning("CustomIterableDataset is restarting its data iteration.")
                self._sample_idx = 0

    def reset(self):
        self._sample_idx = 0

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

class DistributedDataLoader(DataLoader):
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int,
        seq_len: int,
        world_size: int,
        rank: int,
        device_type: str,
        num_workers: int = 0,
        shuffle: bool = False
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        super().__init__(
            dataset,
            batch_size=None,  # We've already handled batching in the dataset
            num_workers=num_workers,
            shuffle=shuffle,
        )
        self.world_size = world_size
        self.rank = rank
        self.device_type = device_type

    def __iter__(self):
        for batch in self.dataset:
            yield {k: v.to(self.device_type) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        yield "end"

    def reset(self):
        self.dataset.reset()

    def state_dict(self) -> Dict[str, Any]:
        return self.dataset.state_dict() if self.dataset else {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.dataset:
            self.dataset.load_state_dict(state_dict)

class DummyDataset(IterableDataset):
    def __iter__(self):
        return iter([])

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

def build_custom_data_loader(
    data_dir: str,
    dataset: str,
    split: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    device_type: str,
    num_workers: int = 0,
    shuffle: bool = False,
    mode: str = "align",
    packing: bool = False,
) -> 'DistributedDataLoader':
    if dataset == "mmlu":
        dataset = MMLUDataset(data_dir, split, tokenizer)
    elif dataset == "hh":
        dataset = HHDataset(
            tokenizer,
            split=split,
            seq_len=seq_len,
            batch_size=batch_size,
            mode=mode,
            packing=packing,
            world_size=world_size,
            rank=rank
        )
    else:
        raise ValueError(f"Unsupported custom dataset: {dataset}")

    return DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        world_size=world_size,
        rank=rank,
        device_type=device_type,
        num_workers=num_workers,
        shuffle=shuffle
    )