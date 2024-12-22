import os
import sys
import torch
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torchtitan.datasets.mmlu_dataset import MMLUDataset, build_mmlu_data_loader
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.config_manager import JobConfig


@pytest.fixture
def job_config():
    config = JobConfig()
    config.parse_args(['--model.tokenizer_path', 'torchtitan/datasets/tokenizer/tokenizer.model'])
    return config


def test_mmlu_dataset(job_config):
    tokenizer = build_tokenizer(job_config.model.name, job_config.model.tokenizer_path)

    dataset = MMLUDataset("torchtitan/datasets/mmlu", "dev", tokenizer, max_seq_len=512)

    item = dataset[0]
    assert "input_ids" in item and "labels" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)

    data_loader = build_mmlu_data_loader(
        "torchtitan/datasets/mmlu", "dev", tokenizer, batch_size=4, max_seq_len=512
    )
    batch = next(iter(data_loader))
    assert "input_ids" in batch and "labels" in batch
    assert batch["input_ids"].shape[0] == 4

    print("Dataset tests passed!")


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, self.vocab_size)


def compute_loss(logits, labels, tokenizer):
    choice_logits = logits[:, -1, tokenizer.encode(" ABCD")]
    true_labels = labels[:, -1] - tokenizer.encode(" A")[0]
    return torch.nn.functional.cross_entropy(choice_logits, true_labels)


def test_training_objective(job_config):
    tokenizer = build_tokenizer(job_config.model.name, job_config.model.tokenizer_path)
    data_loader = build_mmlu_data_loader(
        "torchtitan/datasets/mmlu", "dev", tokenizer, batch_size=4, max_seq_len=512
    )
    model = DummyModel(tokenizer.n_words)

    batch = next(iter(data_loader))
    input_ids, labels = batch["input_ids"], batch["labels"]

    logits = model(input_ids)
    loss = compute_loss(logits, labels, tokenizer)

    assert not torch.isnan(loss) and not torch.isinf(loss)
    print(f"Training objective test passed. Loss: {loss.item()}")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args(['--model.tokenizer_path', 'torchtitan/datasets/tokenizer/tokenizer.model'])
    test_mmlu_dataset(config)
    test_training_objective(config)