import argparse
import os
import sys

import pydevd_pycharm
import torch
import numpy as np
from pathlib import Path

from torch.distributed.tensor.experimental._attention import context_parallel_unshard

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.unit_tests.test_utils import run_torchrun
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.utils import device_type, gather_dtensor, set_default_dtype
from torchtitan.models.reference_model import build_reference_model
from torchtitan.models.llama import attention_utils
from torchtitan.datasets.hh_dataset import build_hh_data_loader
from torchtitan import utils, state
from torchtitan.parallelisms import context

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
    parser.add_argument("--trace_dir", type=str, help="Directory to save attention traces")
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
    job_config.evaluation.batch_size = 4
    job_config.experimental.pipeline_parallel_microbatches = 1
    job_config.training.batch_size = 1
    job_config.training.dataset_mode = "sft"
    job_config.training.dataset_packing = True
    job_config.training.mixed_precision_param = "bfloat16"
    return job_config

def properly_gather_logits(logits, world_mesh, cp_enabled):
    """
    Properly gather distributed logits, handling context parallelism specially

    Args:
        logits: The output logits tensor, potentially distributed
        world_mesh: The device mesh for distribution
        cp_enabled: Whether context parallelism is enabled

    Returns:
        Fully gathered logits tensor
    """
    if logits is None:
        return None

    # First check if this is a DTensor (for any parallelism)
    is_dtensor = isinstance(logits, torch.distributed.tensor.DTensor)

    # Special handling for context parallelism
    if cp_enabled:
        try:
            # If using CP, we need to use context_parallel_unshard
            # Context parallelism shards along sequence dimension (dim=1)
            if is_dtensor:
                # First make it local - this preserves the sharding
                local_logits = logits.to_local()
                # Then unshard using CP-specific function
                return context_parallel_unshard(world_mesh["cp"], [local_logits], [1])[0]
            else:
                # If somehow already local but CP is enabled, we should still unshard
                return context_parallel_unshard(world_mesh["cp"], [logits], [1])[0]
        except Exception as e:
            logger.warning(f"Error using context_parallel_unshard: {e}. Falling back to standard gather.")
            # Fall back to standard gather
            pass

    # For other forms of parallelism or as fallback
    if is_dtensor:
        return gather_dtensor(logits, world_mesh)

    return logits


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
        world_mesh = model.device_mesh
        cp_mesh = world_mesh["cp"]

        # Create CP context with appropriate buffers
        if hasattr(model, 'model_parts'):
            # Initialize buffers, sequence dimensions, and no-restore buffers with input_ids
            cp_buffers = [input_ids]
            cp_seq_dims = [1]
            cp_no_restore_buffers = {input_ids}

            # Only add attention_mask if it is not None
            if attention_mask is not None:
                cp_buffers.append(attention_mask)
                cp_seq_dims.append(2)
                cp_no_restore_buffers.add(attention_mask)

            # Append the freqs_cis from each model part
            cp_buffers.extend([m.freqs_cis for m in model.model_parts])
            cp_seq_dims.extend([0 for _ in model.model_parts])

            optional_context_parallel_ctx = utils.create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=cp_seq_dims,
                cp_no_restore_buffers=cp_no_restore_buffers,
                cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
            )

    # Set state tensors (common for both cases)
    state.set_state_tensors(
        attention_mask=attention_mask,
        batch_size=job_config.training.batch_size,
        n_microbatches=1  # Always 1 for evaluation
    )

    # When using CP, avoid passing the mask twice
    if cp_enabled:
        forward_mask = None
    else:
        forward_mask = attention_mask

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
            logits = model(input_ids, mask=forward_mask)
        end_time.record()

        # Wait for CUDA kernels to finish
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

    # Gather the distributed logits with proper CP handling
    gathered_logits = None
    if logits is not None:
        gathered_logits = properly_gather_logits(logits, model.device_mesh, cp_enabled)

    # Log original and gathered shapes for debugging
    if logits is not None:
        logger.info(f"Original logits shape: {logits.shape}")
        if gathered_logits is not None:
            logger.info(f"Gathered logits shape: {gathered_logits.shape}")

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

    if args.trace_attn:
        if cp_enabled:
            context.disable_attention_tracing()
        else:
            attention_utils.disable_attention_tracing()

    # Clean up
    torch.distributed.barrier()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()


def main():
    args = parse_args()
    if args.run_test:
        with set_default_dtype(torch.float32):
            run_forward_pass()
    else:
        run_torchrun(__file__, args)


if __name__ == "__main__":
    main()