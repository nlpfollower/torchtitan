import argparse
import os
import sys

import pydevd_pycharm
import torch
import numpy as np
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.unit_tests.test_utils import run_torchrun
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.utils import get_device_info, device_type, gather_dtensor
from torchtitan.models.reference_model import build_reference_model
from torchtitan.datasets.hh_dataset import build_hh_data_loader
from torchtitan import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Test context parallelism masking")
    parser.add_argument("--ngpu", type=int, default=int(os.environ.get("NGPU", 1)))
    parser.add_argument("--dp_shard", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output logits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def setup_job_config(args):
    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "3B"
    job_config.model.tokenizer_path = "models/Llama3.2-3B-Instruct/tokenizer.model"
    job_config.training.seq_len = 8192  # Smaller sequence length for testing
    job_config.training.deterministic = True
    job_config.checkpoint.enable_checkpoint = True
    job_config.job.dump_folder = "outputs"
    job_config.checkpoint.folder = "checkpoint"
    job_config.reference_model.checkpoint_path = args.checkpoint_path
    job_config.reference_model.data_parallel_shard_degree = args.dp_shard
    job_config.reference_model.tensor_parallel_degree = args.tp
    job_config.reference_model.pipeline_parallel_degree = args.pp
    job_config.experimental.context_parallel_degree = args.cp
    job_config.experimental.context_parallel_rotate_method = "allgather"
    job_config.experimental.pipeline_parallel_schedule = "GPipe"
    job_config.training.dataset_type = "custom"
    job_config.training.dataset = "hh"
    job_config.training.use_attention_mask = True
    job_config.evaluation.batch_size = 2
    job_config.experimental.pipeline_parallel_microbatches = 1
    job_config.training.batch_size = 2
    job_config.training.dataset_mode = "sft"
    job_config.training.dataset_packing = True
    return job_config


def run_forward_pass():
    """Run a single forward pass and save the output logits"""
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    init_logger()
    logger.info(f"Torch previous num threads: {torch.get_num_threads()}")
    num_threads = os.cpu_count()  # Set to the number of available CPU cores
    num_threads_per_rank = max(1, num_threads // min(world_size, 8))
    torch.set_num_threads(num_threads_per_rank)
    logger.info(f"Torch new num threads: {torch.get_num_threads()}")

    if rank == 0:
        print("Hello from rank 0")
        pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)

    init_logger()
    args = parse_args()
    job_config = setup_job_config(args)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up device
    device = torch.device(f"{device_type}:{rank}")
    torch.cuda.set_device(device)

    # Build tokenizer
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)

    # Create a single test batch from the dataset
    data_loader = build_hh_data_loader(
        tokenizer,
        job_config.evaluation.batch_size,
        job_config.training.seq_len,
        split="test",
        mode=job_config.training.dataset_mode,
        packing=job_config.training.dataset_packing
    )

    # Build the model - reference_model handles parallelism internally
    model = build_reference_model(job_config, tokenizer)

    # Get a batch from the dataset
    batch = next(iter(data_loader))
    if batch == "end":
        logger.error("Dataset is empty, cannot proceed")
        sys.exit(1)

    input_ids = batch['input_ids'].to(device_type)
    attention_mask = None
    if "attention_mask" in batch:
        attention_mask = batch['attention_mask'].to(device_type)

    # Set up context parallelism context manager if CP is enabled
    cp_enabled = job_config.experimental.context_parallel_degree > 1
    optional_context_parallel_ctx = None

    if cp_enabled:
        # Extract necessary components for CP context
        world_mesh = model.device_mesh

        if hasattr(model, 'model_parts'):
            # Create CP context with appropriate buffers
            optional_context_parallel_ctx = utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[input_ids] + [m.freqs_cis for m in model.model_parts],
                cp_seq_dims=[1] + [0 for _ in model.model_parts],
                cp_no_restore_buffers={input_ids},
                cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
            )

    # Create context manager
    train_context = utils.get_train_context(False, False)

    # Run the forward pass
    model.eval()
    with torch.no_grad():
        # Record the start time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        # Use the context manager if CP is enabled
        with train_context(optional_context_parallel_ctx):
            logits = model(input_ids, mask=attention_mask)
        end_time.record()

        # Wait for CUDA kernels to finish
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

    # Gather the distributed logits
    gathered_logits = None
    if logits is not None:
        gathered_logits = gather_dtensor(logits, model.device_mesh)

    # Make sure the output directory exists
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Forward pass completed in {elapsed_time:.3f}s")

    torch.distributed.barrier()

    # Save the logits
    if rank == 0 and gathered_logits is not None:
        # Create a filename that indicates the parallelism configuration
        filename = f"logits_dp{args.dp_shard}_cp{args.cp}_tp{args.tp}_pp{args.pp}.pt"
        output_path = output_dir / filename

        # Save the logits and metadata
        metadata = {
            "shape": gathered_logits.shape,
            "parallel_config": {
                "dp_shard": args.dp_shard,
                "cp": args.cp,
                "tp": args.tp,
                "pp": args.pp
            },
            "elapsed_time": elapsed_time,
        }

        torch.save({
            "logits": gathered_logits,
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
            "metadata": metadata
        }, output_path)

        logger.info(f"Saved logits to {output_path}")

    # Clean up
    torch.distributed.barrier()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


def main():
    args = parse_args()
    if args.run_test:
        run_forward_pass()
    else:
        run_torchrun(__file__, args)


if __name__ == "__main__":
    main()