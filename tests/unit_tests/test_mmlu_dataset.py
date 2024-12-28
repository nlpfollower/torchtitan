import json
import os
import sys
import logging
from pathlib import Path

import pydevd_pycharm
import torch
from torch.utils.data import DataLoader
import math
from torchtitan.objective import Objective

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torchtitan.datasets.mmlu_dataset import MMLUDataset, build_mmlu_data_loader
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.models.llama.model import Transformer, precompute_freqs_cis
from torchtitan.models.llama import llama3_configs

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_config():
    logger.info("Setting up test configuration")
    config = JobConfig()
    config.parse_args(['--model.tokenizer_path', 'models/Llama3.2-3B-Instruct/tokenizer.model'])
    return config


def test_mmlu_dataset():
    config = setup_config()
    logger.info(f"Testing MMLU dataset with tokenizer path: {config.model.tokenizer_path}")

    try:
        tokenizer = build_tokenizer("llama", config.model.tokenizer_path)
        dataset = MMLUDataset("torchtitan/datasets/mmlu", "dev", tokenizer)

        assert len(dataset) > 0, "Dataset is empty"
        logger.info(f"Dataset size: {len(dataset)}")

        item = dataset[0]
        assert all(
            key in item for key in ["input_ids", "labels"]), "Dataset item is missing required keys"
        assert all(isinstance(item[key], torch.Tensor) for key in
                   ["input_ids", "labels"]), "Dataset item values are not torch.Tensor"

        logger.info(
            f"Sample item shapes: input_ids {item['input_ids'].shape}, labels {item['labels'].shape}")

        data_loader = build_mmlu_data_loader(
            "torchtitan/datasets/mmlu", "dev", tokenizer, batch_size=4
        )

        batch = next(iter(data_loader))
        assert all(key in batch for key in ["input_ids", "labels"]), "Batch is missing required keys"
        assert batch["input_ids"].shape[0] == 4, f"Incorrect batch size: {batch['input_ids'].shape[0]}"

        logger.info(
            f"Batch shapes: input_ids {batch['input_ids'].shape}, labels {batch['labels'].shape}")
        logger.info("MMLU Dataset and DataLoader tests passed successfully")

    except Exception as e:
        logger.error(f"Error in test_mmlu_dataset: {str(e)}", exc_info=True)
        raise

def test_training_objective():
    config = setup_config()
    logger.info("Testing training objective")

    try:
        tokenizer = build_tokenizer("llama", config.model.tokenizer_path)
        data_loader = build_mmlu_data_loader(
            "torchtitan/datasets/mmlu", "dev", tokenizer, batch_size=4
        )

        batch = next(iter(data_loader))
        input_ids, labels = batch["input_ids"], batch["labels"]

        # Test 1: Random logits (as before)
        batch_size, seq_len = input_ids.shape
        random_logits = torch.randn(batch_size, seq_len, tokenizer.n_words)
        random_loss = Objective.classification_loss(random_logits, labels)
        logger.info(f"Random logits loss: {random_loss.item()}")

        # Test 2: Perfect prediction logits
        perfect_logits = create_perfect_logits(labels, tokenizer.n_words)
        perfect_loss = Objective.classification_loss(perfect_logits, labels)
        logger.info(f"Perfect prediction loss: {perfect_loss.item()}")

        # Validate perfect prediction loss
        assert torch.isclose(perfect_loss, torch.tensor(0.0), rtol=1e-5), \
            f"Perfect prediction loss {perfect_loss.item()} doesn't match expected 0"

        logger.info("All training objective tests passed successfully.")

    except Exception as e:
        logger.error(f"Error in test_training_objective: {str(e)}", exc_info=True)
        raise

def create_perfect_logits(labels, vocab_size):
    perfect_logits = torch.full((labels.size(0), labels.size(1), vocab_size), -1e9, dtype=torch.float)
    valid_positions = labels[:, 1:] != -100
    perfect_logits[:, :-1][valid_positions] = torch.nn.functional.one_hot(labels[:, 1:][valid_positions], num_classes=vocab_size).float() * 100.0
    return perfect_logits

def calculate_expected_perfect_loss(shifted_labels):
    valid_labels = shifted_labels[shifted_labels != -100]
    n_valid = valid_labels.numel()
    return torch.tensor(0.0) if n_valid == 0 else torch.tensor(-math.log(1.0)) * n_valid

def test_model_inference():
    config = setup_config()
    logger.info("Testing model inference on MMLU dataset")
    max_seq_len = 2048

    try:
        tokenizer = build_tokenizer("llama", config.model.tokenizer_path)
        data_loader = build_mmlu_data_loader(
            "torchtitan/datasets/mmlu", "dev", tokenizer, batch_size=4
        )

        # Load the model
        model_path = "models/Llama3.2-3B-Instruct/consolidated.00.pth"
        model_args = llama3_configs["3B"]
        model_args.vocab_size = tokenizer.n_words


        input_dir = Path("models/Llama3.2-3B-Instruct")
        with open(input_dir / "params.json", "r") as f:
            params = json.load(f)
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads

        checkpoint_list = sorted([file for file in input_dir.rglob("*.pth")])
        logger.info(
            f"Loading original Llama weights from {[ckpt.name for ckpt in checkpoint_list]}"
        )

        model = Transformer(model_args)
        for ckpt in checkpoint_list:
            shard = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
            shard['freqs_cis'] = precompute_freqs_cis(dims_per_head, max_seq_len, params.get("rope_theta", 500000),)
            model.load_state_dict(shard)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        total_loss = 0
        num_samples = 5

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_samples:
                    break

                input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(input_ids)

                loss = Objective.classification_loss(logits, labels, debug=True)
                total_loss += loss.item()

                logger.info(f"Sample {i + 1} loss: {loss.item()}")

        avg_loss = total_loss / num_samples
        logger.info(f"Average loss over {num_samples} samples: {avg_loss}")

        perplexity = torch.exp(torch.tensor(avg_loss))
        logger.info(f"Perplexity: {perplexity.item()}")

    except Exception as e:
        logger.error(f"Error in test_model_inference: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    logger.info("Starting MMLU dataset tests")
    test_mmlu_dataset()
    test_training_objective()
    test_model_inference()
    logger.info("All tests completed successfully")