from typing import List, Dict, Any, Optional

import torch
from click.core import batch
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed import broadcast
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

class DistributedDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Optional[IterableDataset],
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
            dataset if rank == 0 else DummyDataset(),
            batch_size=batch_size,
            num_workers=num_workers if rank == 0 else 0,
            shuffle=shuffle,
        )
        self.world_size = world_size
        self.rank = rank
        self.device_type = device_type

    def __iter__(self):
        while True:
            if self.rank == 0:
                try:
                    batch = next(iter(self.dataset))
                    input_ids = batch['input_ids'].to(self.device_type)
                    labels = batch['labels'].to(self.device_type)
                    document_ids = batch.get('document_ids')
                    if document_ids is not None:
                        document_ids = document_ids.to(self.device_type)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device_type)
                except StopIteration:
                    break
            else:
                input_ids = torch.empty((self.batch_size, self.seq_len), dtype=torch.long).to(self.device_type)
                labels = torch.empty_like(input_ids).to(self.device_type)
                document_ids = torch.empty_like(input_ids).to(self.device_type)
                attention_mask = torch.empty((self.batch_size, 1, self.seq_len, self.seq_len), dtype=torch.bool).to(self.device_type)

            broadcast(input_ids, src=0)
            broadcast(labels, src=0)
            broadcast(document_ids, src=0)
            broadcast(attention_mask, src=0)

            yield {
                'input_ids': input_ids,
                'labels': labels,
                'document_ids': document_ids,
                'attention_mask': attention_mask
            }

    def state_dict(self) -> Dict[str, Any]:
        return self.dataset.state_dict() if self.dataset else {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.dataset:
            self.dataset.load_state_dict(state_dict)

class DummyDataset(IterableDataset):
    def __iter__(self):
        return iter([])

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
) -> 'DistributedDataLoader':
    if rank == 0:
        if dataset == "mmlu":
            dataset = MMLUDataset(data_dir, split, tokenizer)
        elif dataset == "hh":
            dataset = HHDataset(tokenizer, split=split, seq_len=seq_len, batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported custom dataset: {dataset}")
    else:
        dataset = None

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