import os
import pandas as pd
import torch
import csv
from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from typing import List, Dict, Any, Tuple, Iterator, Optional
from tqdm import tqdm

from torchtitan.datasets.tokenizer import Tokenizer
from llama_models.llama3.api import ChatFormat, ToolPromptFormat, RawMessage
from torchtitan.models.llama.attention_utils import create_document_causal_mask


class FileDataset(IterableDataset):
    def __init__(self,
                 tokenizer: Tokenizer,
                 file_path: str,
                 split: str = "train",
                 seq_len: int = 2048,
                 batch_size: int = 1,
                 packing: bool = False,
                 world_size: int = 1,
                 rank: int = 0,
                 system_prompt: str = "You are a helpful assistant.",
                 mode: str = "sft"):
        """
        Dataset for loading and processing SFT data from a CSV file.

        Args:
            tokenizer: The tokenizer to use for encoding
            file_path: Path to the CSV file containing the dataset
            split: Split to use (only used for compatibility with the existing framework)
            seq_len: Maximum sequence length
            batch_size: Batch size
            packing: Whether to pack multiple samples into one sequence
            world_size: Number of workers in distributed setting
            rank: Worker rank in distributed setting
            system_prompt: System prompt to prepend to each conversation
            mode: Training mode, currently only "sft" is supported
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.chat_format = ChatFormat(tokenizer)
        self.packing = packing
        self.system_prompt = system_prompt
        self.mode = mode.lower()

        # Check if mode is supported
        if self.mode != "sft":
            raise ValueError(f"Mode '{mode}' is not supported. Currently only 'sft' mode is available.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        print(f'Loading SFT dataset from CSV file: {file_path}')

        # Load the CSV file using pandas
        try:
            df = pd.read_csv(file_path)
            # Ensure the CSV has the expected columns
            if 'prompt' not in df.columns or 'response' not in df.columns:
                raise ValueError("CSV file must contain 'prompt' and 'response' columns")

            # Convert to Hugging Face dataset for distributed splitting
            full_dataset = Dataset.from_pandas(df)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        print(f'Dataset loaded with {len(full_dataset)} samples. Processing...')

        # If dataset is smaller than world_size, repeat samples to ensure at least world_size samples
        if len(full_dataset) < world_size:
            print(f'Dataset size ({len(full_dataset)}) is smaller than world_size ({world_size}). Repeating samples...')

            # We need at least world_size samples total
            samples_needed = world_size - len(full_dataset)

            # Create a list of repeated dataframes (single rows)
            repeated_dfs = [df]  # Start with original dataframe
            for i in range(samples_needed):
                # Use iloc[[i]] to get a DataFrame instead of Series
                repeated_dfs.append(df.iloc[[i % len(df)]])

            # Combine all dataframes
            repeated_df = pd.concat(repeated_dfs, ignore_index=True)

            full_dataset = Dataset.from_pandas(repeated_df)
            print(f'Dataset expanded to {len(full_dataset)} samples')

        # Split the dataset by node using Hugging Face's split_dataset_by_node
        self.dataset = split_dataset_by_node(full_dataset, rank, world_size)

        self.batches = self.load_and_batch_data()
        self._sample_idx = 0

    def load_and_batch_data(self) -> List[Dict[str, torch.Tensor]]:
        """Process the dataset into batches ready for training"""
        batches = []
        current_batch = []
        current_sample = {"input_ids": [], "labels": [], "document_ids": []}
        current_length = 0
        current_doc_id = 0

        for i in tqdm(range(len(self.dataset)), desc='Processing CSV data', total=len(self.dataset)):
            row = self.dataset[i]
            # Process the sample (prompt and response)
            sample = self.process_sample(row['prompt'], row['response'])

            # Ensure sample isn't too long
            if len(sample['input_ids']) > self.seq_len:
                print(
                    f"Warning: Sample exceeds max sequence length ({len(sample['input_ids'])} > {self.seq_len}), truncating")
                for key in ['input_ids', 'labels', 'document_ids']:
                    sample[key] = sample[key][:self.seq_len]

            if self.packing:
                # Packing mode: combine multiple samples into a sequence up to seq_len
                if current_length + len(sample['input_ids']) > self.seq_len:
                    # Current sample is full, finalize it and start a new one
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
                current_doc_id += 1
            else:
                # Non-packing mode: each sample is individually processed
                self.pad_sample(sample, self.seq_len)
                current_batch.append(sample)

                if len(current_batch) == self.batch_size:
                    batches.append(self.prepare_batch(current_batch))
                    current_batch = []

        # Handle any remaining data in packing mode
        if self.packing and current_sample["input_ids"]:
            self.pad_sample(current_sample, self.seq_len)
            current_batch.append(current_sample)

        # Handle any remaining batch (for both modes)
        if current_batch:
            while len(current_batch) < self.batch_size:
                current_batch.append(self.create_empty_sample())
            batches.append(self.prepare_batch(current_batch))

        return batches

    def process_sample(self, prompt: str, response: str) -> Dict[str, List[int]]:
        """Process a prompt-response pair into tokens for SFT training"""
        # Handle None values
        if prompt is None or response is None:
            print(f"Warning: Found None value in data (prompt: {prompt}, response: {response}). Skipping sample.")
            return {
                "input_ids": [self.tokenizer.pad_id],
                "labels": [-100],
                "document_ids": [-1]
            }

        # Convert to string and strip
        prompt = str(prompt).strip()
        response = str(response).strip()

        # Skip empty samples
        if not prompt or not response:
            print(f"Warning: Found empty prompt or response. Skipping sample.")
            return {
                "input_ids": [self.tokenizer.pad_id],
                "labels": [-100],
                "document_ids": [-1]
            }

        # Create system message
        system_message = RawMessage(role="system", content=self.system_prompt)

        # Create user message from prompt
        user_message = RawMessage(role="user", content=prompt)

        # Create assistant message from response
        assistant_message = RawMessage(role="assistant", content=response)

        # Tokenize messages
        prompt_tokens = [self.tokenizer.bos_id]

        # Add system and user messages to prompt tokens
        for message in [system_message, user_message]:
            toks, _ = self.chat_format.encode_message(message, tool_prompt_format=ToolPromptFormat.json)
            prompt_tokens.extend(toks)

        # Tokenize the assistant's response
        response_tokens, _ = self.chat_format.encode_message(assistant_message,
                                                             tool_prompt_format=ToolPromptFormat.json)
        response_tokens.append(self.tokenizer.eos_id)

        # For SFT, we want to predict only the response
        input_ids = prompt_tokens + response_tokens
        labels = [-100] * len(prompt_tokens) + response_tokens
        document_ids = [0] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "document_ids": document_ids
        }

    def pad_sample(self, sample: Dict[str, List[int]], target_length: int):
        """Pad sample to target length if needed"""
        pad_length = max(0, target_length - len(sample['input_ids']))
        sample['input_ids'].extend([self.tokenizer.pad_id] * pad_length)
        sample['labels'].extend([-100] * pad_length)
        sample['document_ids'].extend([-1] * pad_length)

    def prepare_batch(self, samples: List[Dict[str, List[int]]]) -> Dict[str, any]:
        """Convert a list of samples into a batch tensor"""
        batch = {
            key: torch.tensor([sample[key] for sample in samples], dtype=torch.long)
            for key in ['input_ids', 'labels', 'document_ids']
        }

        # Create attention mask only in packing mode
        if self.packing:
            batch['attention_mask'] = create_document_causal_mask(batch['document_ids'])

        return batch

    def create_empty_sample(self):
        """Create an empty sample filled with padding tokens"""
        return {
            "input_ids": [self.tokenizer.pad_id] * self.seq_len,
            "labels": [-100] * self.seq_len,
            "document_ids": [-1] * self.seq_len,
        }

    def __len__(self):
        return len(self.batches)

    def __iter__(self) -> Iterator[Dict[str, any]]:
        for i, batch in enumerate(self.batches[self._sample_idx:]):
            self._sample_idx += 1
            yield batch

    def reset(self):
        """Reset the dataset iteration"""
        self._sample_idx = 0

    def state_dict(self):
        """Get the state dict for checkpointing"""
        return {"sample_idx": self._sample_idx}

    def load_state_dict(self, state_dict):
        """Load the state from a checkpoint"""
        self._sample_idx = state_dict["sample_idx"]


def build_file_data_loader(
        tokenizer: Tokenizer,
        file_path: str,
        batch_size: int,
        seq_len: int = 2048,
        split: str = "train",
        num_workers: int = 0,
        packing: bool = False,
        world_size: int = 1,
        rank: int = 0,
        system_prompt: str = "You are a helpful assistant.",
        mode: str = "sft",
) -> DataLoader:
    """Build a DataLoader for the CSV file dataset with specified configuration"""
    dataset = FileDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        split=split,
        seq_len=seq_len,
        batch_size=batch_size,
        packing=packing,
        world_size=world_size,
        rank=rank,
        system_prompt=system_prompt,
        mode=mode
    )
    return DataLoader(
        dataset,
        batch_size=None,  # Batching is handled in the dataset
        num_workers=num_workers,
        shuffle=False,  # Ensure deterministic ordering
    )