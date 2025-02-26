import os

import pydevd_pycharm
import torch
import torch.distributed as dist
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.datasets.hh_dataset import build_hh_data_loader
from torchtitan.models.reference_model import build_reference_model
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import get_device_info, set_determinism
from torchtitan.logging import logger, init_logger
from torchtitan.models.llama.attention_utils import create_block_document_causal_mask
from torchtitan.objective import Objective
import torch.nn.functional as F

def setup_environment():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="nccl")
    device_type, _ = get_device_info()
    device = torch.device(f"{device_type}:0")
    set_determinism(None, device, seed=42)
    return device, device_type


def create_job_config():
    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "3B"
    job_config.model.tokenizer_path = "models/Llama3.2-3B-Instruct/tokenizer.model"
    job_config.job.dump_folder = "outputs"
    job_config.checkpoint.folder = "checkpoint"
    job_config.checkpoint.enable_checkpoint = True
    job_config.training.seq_len = 8192
    job_config.reference_model.checkpoint_path = "outputs/checkpoint/step-0"
    job_config.reference_model.data_parallel_shard_degree = 1
    job_config.reference_model.tensor_parallel_degree = 1
    job_config.reference_model.pipeline_parallel_degree = 1
    return job_config


def test_dpo_objective(max_seq_length: int, batch_size: int):
    device, device_type = setup_environment()
    job_config = create_job_config()
    job_config.model.attention = "sdpa"
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    model = build_reference_model(job_config, tokenizer)

    # Build dataloader and get a single batch
    dataloader = build_hh_data_loader(tokenizer, batch_size, max_seq_length, split="train")
    batch = next(iter(dataloader))

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    document_ids = batch['document_ids'].to(device)
    attention_mask = create_block_document_causal_mask(document_ids).to(device)

    # Forward pass for "policy" model
    with torch.no_grad():
        policy_logits = model(input_ids, attention_mask)

    # Forward pass for "reference" model (same as policy)
    with torch.no_grad():
        reference_logits = model(input_ids, attention_mask)

    # Compute DPO loss
    loss_fn = Objective.get_loss_function("dpo_with_packing")
    loss = loss_fn(policy_logits, labels, reference_logits, document_ids, beta=0.1)

    logger.info(f"DPO Loss: {loss.item()}")
    dummy_dpo = F.logsigmoid(torch.tensor(0.0, device=device, dtype=loss.dtype))
    assert torch.isclose(loss.abs(), dummy_dpo.abs(), atol=1e-6), f"Loss {loss.item()} is not close to 0"


def test_classification_loss_with_packing(max_seq_length: int, batch_size: int):
    device, device_type = setup_environment()
    job_config = create_job_config()
    job_config.model.attention = "sdpa"
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    model = build_reference_model(job_config, tokenizer)

    # Build dataloader and get a single batch (SFT mode)
    dataloader = build_hh_data_loader(tokenizer, batch_size, max_seq_length, split="train", mode="sft")
    batch = next(iter(dataloader))

    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    document_ids = batch['document_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward pass for packed sample
    with torch.no_grad():
        packed_logits = model(input_ids, attention_mask)

    # Compute packed loss
    packed_loss = Objective.classification_loss_with_packing(packed_logits, labels, document_ids)
    logger.info(f"Packed Loss: {packed_loss.item():.4f}")

    # Compute individual losses
    individual_losses = []
    for i in range(input_ids.shape[0]):
        batch_losses = []
        for doc_id in torch.unique(document_ids[i]):
            if doc_id == -1:  # Skip padding
                continue
            doc_mask = (document_ids[i] == doc_id)
            doc_input_ids = input_ids[i, doc_mask]
            doc_labels = labels[i, doc_mask].unsqueeze(0)

            with torch.no_grad():
                doc_logits = model(doc_input_ids.unsqueeze(0), None)

            doc_loss = Objective.classification_loss(doc_logits, doc_labels)
            batch_losses.append(doc_loss)
            logger.info(f"Individual - Batch {i}, Document {doc_id.item()}: Loss = {doc_loss.item():.4f}")

        batch_mean_loss = torch.stack(batch_losses).mean()
        individual_losses.append(batch_mean_loss)
        logger.info(f"Individual - Batch {i}: Mean Loss = {batch_mean_loss.item():.4f}")

    mean_individual_loss = torch.stack(individual_losses).mean()
    logger.info(f"Mean Individual Loss: {mean_individual_loss.item():.4f}")

    # Assert that the losses are close
    assert torch.isclose(packed_loss, mean_individual_loss, rtol=1e-4, atol=1e-4), \
        f"Packed loss {packed_loss.item():.4f} is not close to mean individual loss {mean_individual_loss.item():.4f}"

    logger.info("Classification loss with packing test passed successfully!")

if __name__ == "__main__":
    # pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    init_logger()
    logger.info("Starting DPO objective test")
    test_classification_loss_with_packing(max_seq_length=4096, batch_size=2)
    # test_dpo_objective(max_seq_length=4096, batch_size=2)
    # dist.destroy_process_group()