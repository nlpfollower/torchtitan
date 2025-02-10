import re

import torch
from llama_models.llama3.api import ChatFormat, ToolPromptFormat, RawMessage, Role
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Any, Tuple, Iterator
from datasets import load_dataset
from torchtitan.datasets.tokenizer import Tokenizer
from collections import defaultdict
from tqdm import tqdm

def packed_document_causal_mask(document_ids: torch.Tensor):
    batch_size, max_seq_len = document_ids.shape
    device = document_ids.device

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        non_padding = document_ids[b, q_idx] != -1
        return causal_mask & document_mask & non_padding

    return create_block_mask(
        mask_mod,
        batch_size,
        None,
        max_seq_len,
        max_seq_len,
        device=device,
    )

def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

class HHDataset(IterableDataset):
    def __init__(self, tokenizer: Tokenizer, split: str = "train", seq_len: int = 2048, batch_size: int = 1, cache_dir: str = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chat_format = ChatFormat(tokenizer)
        self.batches = self.load_and_batch_data(split, cache_dir)

    def load_and_batch_data(self, split: str, cache_dir: str) -> List[Dict[str, torch.Tensor]]:
        print(f'Loading HH dataset ({split} split) from Huggingface...')
        dataset = load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
        print('Dataset loaded. Processing...')

        batches = []
        current_batch = []
        current_sample = {"input_ids": [], "labels": [], "document_ids": []}
        current_length = 0
        current_doc_id = 0

        for row in tqdm(dataset, desc='Processing HH'):
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
            current_doc_id += 2

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
        input_ids = prompt + chosen + prompt + rejected
        labels = [-100] * len(prompt) + chosen + [-100] * len(prompt) + rejected
        document_ids = [0] * (len(prompt) + len(chosen)) + [1] * (len(prompt) + len(rejected))

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

        return batch

    def create_empty_sample(self):
        return {
            "input_ids": [self.tokenizer.pad_id] * self.seq_len,
            "labels": [-100] * self.seq_len,
            "document_ids": [-1] * self.seq_len
        }

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[Dict[str, any]]:
        return iter(self.batches)

def build_hh_data_loader(
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int = 2048,
    split: str = "train",
    num_workers: int = 4,
    cache_dir: str = None,
) -> DataLoader:
    dataset = HHDataset(tokenizer, split=split, seq_len=seq_len, batch_size=batch_size, cache_dir=cache_dir)
    return DataLoader(
        dataset,
        batch_size=None,  # We've already handled batching in the dataset
        num_workers=num_workers,
        shuffle=False,  # Ensure deterministic ordering
    )