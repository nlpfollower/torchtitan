from torchtitan.logging import logger
import os
import pandas as pd
import torch
from llama_models.llama3.api import ChatFormat, RawMessage
from torch.utils.data import Dataset, DataLoader
from torchtitan.datasets.tokenizer import Tokenizer
from typing import List, Dict

class MMLUDataset(Dataset):
    def __init__(self, data_dir: str, split: str, tokenizer: Tokenizer):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.chat_format = ChatFormat(tokenizer)
        self.data = self.load_data()
        self.max_seq_len = self.calculate_max_seq_len()
        logger.info(f"Initialized MMLUDataset with {len(self.data)} samples and max_seq_len {self.max_seq_len}")

    def load_data(self):
        data = []
        try:
            subjects = [f.split(f"_{self.split}.csv")[0] for f in os.listdir(os.path.join(self.data_dir, self.split)) if
                        f.endswith(f"_{self.split}.csv")]
            for subject in subjects:
                file_path = os.path.join(self.data_dir, self.split, f"{subject}_{self.split}.csv")
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                df = pd.read_csv(file_path, header=None)
                for _, row in df.iterrows():
                    question = row[0]
                    choices = row[1:5].tolist()
                    answer = row[5]
                    data.append((subject, question, choices, answer))
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def calculate_max_seq_len(self):
        max_len = 0
        for subject, question, choices, answer in self.data:
            messages = self.construct_messages(subject, question, choices)
            llm_input = self.chat_format.encode_dialog_prompt(messages)

            # Account for the answer token and EOT token
            sequence_length = len(llm_input.tokens) + 2

            max_len = max(max_len, sequence_length)

        # Add a small buffer for potential variations
        return max_len

    def construct_messages(self, subject: str, question: str, choices: List[str]) -> List[RawMessage]:
        system_message = RawMessage(role="system", content="You are a helpful assistant.")
        user_instruction = (
            "Reply to the question below with a single letter corresponding to the correct answer choice. "
        )
        question_text = f"Subject: {subject}\nQuestion: {question}\n"
        for i, choice in enumerate(choices):
            question_text += f"{chr(65 + i)}. {choice}\n"

        user_message = RawMessage(role="user", content=f"{user_instruction}\n\n{question_text}")
        return [system_message, user_message]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        subject, question, choices, answer = self.data[idx]
        messages = self.construct_messages(subject, question, choices)

        llm_input = self.chat_format.encode_dialog_prompt(messages)
        answer_token = self.tokenizer.encode(answer, bos=False, eos=False)[0]
        eot_token = self.tokenizer.special_tokens["<|end_of_text|>"]

        full_sequence = llm_input.tokens + [answer_token, eot_token]
        attention_mask = [1] * len(full_sequence)

        padding_length = self.max_seq_len - len(full_sequence)
        if padding_length > 0:
            full_sequence += [self.tokenizer.pad_id] * padding_length
            attention_mask += [0] * padding_length

        labels = [-100] * len(llm_input.tokens) + [answer_token, eot_token] + [-100] * padding_length

        return {
            "input_ids": torch.tensor(full_sequence, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def build_mmlu_data_loader(
    data_dir: str,
    split: str,
    tokenizer: Tokenizer,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    dataset = MMLUDataset(data_dir, split, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate_fn, num_workers=num_workers)