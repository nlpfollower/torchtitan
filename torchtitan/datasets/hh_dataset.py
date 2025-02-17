import re

import torch
from datasets.distributed import split_dataset_by_node
from llama_models.llama3.api import ChatFormat, ToolPromptFormat, RawMessage, Role
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Tuple, Iterator
from datasets import load_dataset
from torchtitan.datasets.tokenizer import Tokenizer
from collections import defaultdict
from tqdm import tqdm

from torchtitan.models.llama.attention_utils import create_document_causal_mask


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

class HHDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, split: str = "train", seq_len: int = 2048, batch_size: int = 1,
                 cache_dir: str = None, mode: str = "align",  world_size: int = 1, rank: int = 0):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chat_format = ChatFormat(tokenizer)
        self.mode = mode
        if mode not in ["align", "sft"]:
            raise ValueError("Mode must be either 'align' or 'sft'")
        print(f'Loading HH dataset ({split} split) from Huggingface...')
        full_dataset = load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
        print('Dataset loaded. Processing...')
        self.dataset = split_dataset_by_node(full_dataset, rank, world_size)
        self.batches = self.load_and_batch_data(split, cache_dir)
        self._sample_idx = 0

    def load_and_batch_data(self, split: str, cache_dir: str) -> List[Dict[str, torch.Tensor]]:
        batches = []
        current_batch = []
        current_sample = {"input_ids": [], "labels": [], "document_ids": []}
        current_length = 0
        current_doc_id = 0
        curr = 0

        for row in tqdm(self.dataset, desc='Processing HH'):
            if curr > 10000:
                break
            curr += 1
            prompt_tokens, chosen_tokens, rejected_tokens = self.process_row(row)
            sample = self.process_sample(prompt_tokens, chosen_tokens, rejected_tokens)

            if current_length + len(sample['input_ids']) > self.seq_len:
                # Pad the current sample to seq_len and add it to the current batch
                self.pad_sample(current_sample, self.seq_len)
                current_batch.append(current_sample)

                if len(current_batch) == self.batch_size:
                    batches.append(self.prepare_batch(current_batch))
                    current_batch = []

                current_sample = {"input_ids": [], "labels": [], "document_ids": []}
                current_length = 0
                current_doc_id = 0

            # Extend the current sample with the new data
            current_sample["input_ids"].extend(sample["input_ids"])
            current_sample["labels"].extend(sample["labels"])
            current_sample["document_ids"].extend([id + current_doc_id for id in sample["document_ids"]])

            current_length += len(sample['input_ids'])
            current_doc_id += 1 if self.mode == "sft" else 2

        # Handle any remaining data
        if current_sample["input_ids"]:
            self.pad_sample(current_sample, self.seq_len)
            current_batch.append(current_sample)

        if current_batch:
            while len(current_batch) < self.batch_size:
                current_batch.append(self.create_empty_sample())
            batches.append(self.prepare_batch(current_batch))

        return batches

    def parse_conversation(self, text: str) -> List[RawMessage]:
        parts = re.split(r'(\n\nHuman:|\n\nAssistant:)', text)
        messages = []
        current_role = None
        current_content = ""

        for part in parts:
            if part == "\n\nHuman:":
                if current_role:
                    messages.append(RawMessage(role=current_role, content=current_content.strip()))
                current_role = "user"
                current_content = ""
            elif part == "\n\nAssistant:":
                if current_role:
                    messages.append(RawMessage(role=current_role, content=current_content.strip()))
                current_role = "assistant"
                current_content = ""
            else:
                current_content += part

        if current_role:
            messages.append(RawMessage(role=current_role, content=current_content.strip()))

        return messages

    def process_row(self, row: Dict[str, str]) -> Tuple[List[int], List[int], List[int]]:
        chosen_parts = re.split(r'(\n\nHuman:|\n\nAssistant:)', row['chosen'])
        rejected_parts = re.split(r'(\n\nHuman:|\n\nAssistant:)', row['rejected'])

        # Find shared prefix
        prefix_parts = []
        for c_part, r_part in zip(chosen_parts, rejected_parts):
            if c_part == r_part:
                prefix_parts.append(c_part)
            else:
                break

        if prefix_parts[-1] == "\n\nAssistant:":
            prefix_parts = prefix_parts[:-1]
        prompt = ''.join(prefix_parts)
        chosen = ''.join(chosen_parts[len(prefix_parts):])
        rejected = ''.join(rejected_parts[len(prefix_parts):])

        # Remove " Human:" and " Assistant:"
        prompt = re.sub(r' (Human|Assistant):', '', prompt)
        chosen = re.sub(r' (Human|Assistant):', '', chosen)
        rejected = re.sub(r' (Human|Assistant):', '', rejected)

        def encode_messages(messages):
            tokens = []
            for message in messages:
                toks, _ = self.chat_format.encode_message(message, tool_prompt_format=ToolPromptFormat.json)
                tokens.extend(toks)
            return tokens

        # Tokenize
        prompt_tokens = [self.tokenizer.bos_id]
        prompt_tokens.extend(encode_messages([RawMessage(role="system", content="You are a helpful assistant.")] +
            self.parse_conversation(prompt)))
        chosen_tokens = encode_messages(self.parse_conversation(chosen)) + [self.tokenizer.eos_id]
        rejected_tokens = encode_messages(self.parse_conversation(rejected)) + [self.tokenizer.eos_id]

        return prompt_tokens, chosen_tokens, rejected_tokens

    def process_sample(self, prompt: List[int], chosen: List[int], rejected: List[int]) -> Dict[str, List[int]]:
        if self.mode == "align":
            input_ids = prompt + chosen + prompt + rejected
            labels = [-100] * len(prompt) + chosen + [-100] * len(prompt) + rejected
            document_ids = [0] * (len(prompt) + len(chosen)) + [1] * (len(prompt) + len(rejected))
        else:  # sft mode
            input_ids = prompt + chosen
            labels = [-100] * len(prompt) + chosen
            document_ids = [0] * (len(prompt) + len(chosen))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "document_ids": document_ids
        }

    def pad_sample(self, sample: Dict[str, List[int]], target_length: int):
        pad_length = target_length - len(sample['input_ids'])
        sample['input_ids'].extend([self.tokenizer.pad_id] * pad_length)
        sample['labels'].extend([-100] * pad_length)
        sample['document_ids'].extend([-1] * pad_length)

    def prepare_batch(self, samples: List[Dict[str, List[int]]]) -> Dict[str, any]:
        batch = {
            key: torch.tensor([sample[key] for sample in samples], dtype=torch.long)
            for key in ['input_ids', 'labels', 'document_ids']
        }
        batch['attention_mask'] = create_document_causal_mask(batch['document_ids'])

        return batch

    def create_empty_sample(self):
        return {
            "input_ids": [self.tokenizer.pad_id] * self.seq_len,
            "labels": [-100] * self.seq_len,
            "document_ids": [-1] * self.seq_len,
            "attention_mask": None
        }

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[Dict[str, any]]:
        for i, batch in enumerate(self.batches[self._sample_idx:]):
            self._sample_idx += 1
            yield batch

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

def build_hh_data_loader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int = 2048,
    split: str = "train",
    num_workers: int = 0,
    cache_dir: str = None,
    mode: str = "sft",
) -> DataLoader:
    dataset = HHDataset(tokenizer, split=split, seq_len=seq_len, batch_size=batch_size, cache_dir=cache_dir, mode=mode)
    return DataLoader(
        dataset,
        batch_size=None,  # We've already handled batching in the dataset
        num_workers=num_workers,
        shuffle=False,  # Ensure deterministic ordering
    )