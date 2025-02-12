from typing import List

import torch
from click.core import batch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset
from torchdata.nodes import Stateful

from torchtitan.datasets.hf_datasets import DPAwareDataLoader
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
            for i, sample in enumerate(self.dataset):
                if i % self.world_size == self.rank:
                    self._sample_idx += 1
                    yield sample['input_ids'], sample['labels']

            if not self.infinite:
                logger.warning("CustomIterableDataset has exhausted its data.")
                break
            else:
                logger.warning("CustomIterableDataset is restarting its data iteration.")
                self._sample_idx = 0

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

def build_custom_data_loader(
    data_dir: str,
    dataset: str,
    split: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size: int,
    rank: int,
    num_workers: int = 0,
    shuffle: bool = False,
    infinite: bool = True,
) -> DataLoader:
    if dataset == "mmlu":
        dataset = MMLUDataset(data_dir, split, tokenizer)
    elif dataset == "hh":
        dataset = HHDataset(tokenizer, split=split, seq_len=seq_len, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported custom dataset: {dataset}")

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        shuffle=shuffle,
    )