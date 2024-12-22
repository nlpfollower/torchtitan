import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtitan.datasets.tokenizer import Tokenizer

class MMLUDataset(Dataset):
    def __init__(self, data_dir: str, split: str, tokenizer: Tokenizer, max_seq_len: int):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = self.load_data()

    def load_data(self):
        data = []
        subjects = [f.split(f"_{self.split}.csv")[0] for f in os.listdir(os.path.join(self.data_dir, self.split)) if
                    f.endswith(f"_{self.split}.csv")]
        for subject in subjects:
            df = pd.read_csv(os.path.join(self.data_dir, self.split, f"{subject}_{self.split}.csv"), header=None)
            for _, row in df.iterrows():
                question = row[0]
                choices = row[1:5].tolist()
                answer = row[5]
                data.append((subject, question, choices, answer))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject, question, choices, answer = self.data[idx]
        prompt = f"Subject: {subject}\nQuestion: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"

        input_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        label_ids = self.tokenizer.encode(f" {answer}", bos=False, eos=True)

        input_ids = input_ids[:self.max_seq_len]
        label_ids = label_ids[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def build_mmlu_data_loader(
        data_dir: str,
        split: str,
        tokenizer: Tokenizer,
        batch_size: int,
        max_seq_len: int,
        num_workers: int = 4,
):
    dataset = MMLUDataset(data_dir, split, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)