import argparse
import os
import sys
import time

import pydevd_pycharm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from tests.unit_tests.test_utils import run_torchrun
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import get_device_info, set_determinism
from torchtitan.models.reference_model import build_reference_model
from torchtitan.objective import Objective
from torchtitan.datasets.hh_dataset import build_hh_data_loader
from torchtitan.utils import device_type

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--ngpu", type=int, default=int(os.environ.get("NGPU", 1)))
    parser.add_argument("--dp_shard", type=int, default=1)
    parser.add_argument("--dp_replicate", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--run_test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def setup_job_config(args):
    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "3B"
    # job_config.model.tokenizer_path = "/opt/dlami/nvme/Llama3.1-8B-Instruct/tokenizer.model"
    # job_config.training.seq_len = 8192
    # job_config.training.deterministic = True
    # job_config.checkpoint.enable_checkpoint = True
    # job_config.job.dump_folder = "/opt/dlami/nvme/out"
    # job_config.checkpoint.folder = "checkpoint"
    job_config.model.tokenizer_path = "models/Llama3.2-3B-Instruct/tokenizer.model"
    job_config.training.seq_len = 8192
    job_config.training.deterministic = True
    job_config.checkpoint.enable_checkpoint = True
    job_config.job.dump_folder = "outputs"
    job_config.checkpoint.folder = "checkpoint"
    job_config.reference_model.checkpoint_path = args.checkpoint_path
    job_config.reference_model.data_parallel_shard_degree = args.dp_shard
    job_config.reference_model.tensor_parallel_degree = args.tp
    job_config.reference_model.pipeline_parallel_degree = args.pp
    job_config.experimental.pipeline_parallel_schedule = "ZBVZeroBubble"
    job_config.training.dataset_type = "custom"
    job_config.training.dataset = "hh"
    job_config.training.loss_function = "classification_with_packing"
    job_config.evaluation.batch_size = 2
    job_config.experimental.pipeline_parallel_microbatches = 2
    job_config.training.batch_size = 2
    job_config.evaluation.enabled = True
    job_config.training.dataset_mode = "sft"
    job_config.training.dataset_packing = True
    job_config.evaluation.num_samples = 10
    job_config.evaluation.interval = 30
    return job_config


def evaluate(model, rank, world_size, eval_iter, loss_fn, world_mesh, device_type, job_config):
    model.eval()
    eval_losses = []
    eval_perplexities = []
    num_samples = 0

    is_eval_exhausted = torch.zeros(world_size, dtype=torch.bool, device=device_type)
    with torch.no_grad():
        while True:
            try:
                batch = next(eval_iter)
                if batch == "end":
                    is_eval_exhausted[rank] = True
                torch.distributed.all_reduce(is_eval_exhausted)
                if torch.any(is_eval_exhausted):
                    break
            except StopIteration:
                break

            input_ids = batch['input_ids'].to(device_type)
            labels = batch['labels'].to(device_type)
            document_ids = batch['document_ids'].to(device_type)
            attention_mask = None
            if "attention_mask" in batch:
                attention_mask = batch['attention_mask'].to(device_type)

            logits = model(input_ids, mask=attention_mask)
            loss = torch.zeros(1, dtype=next(model.parameters()).dtype, device=device_type)

            # Compute loss only on last stage
            if logits is not None:
                loss = loss_fn(logits, labels, document_ids=document_ids).reshape(1)

            # Broadcast loss if using pipeline parallel
            if world_mesh.mesh_dim_names is not None and "pp" in world_mesh.mesh_dim_names:
                pp_group = world_mesh["pp"].get_group()
                last_stage_rank = model.stages[0].stage_index_to_group_rank[model.total_stages - 1]

                logger.info(f"Rank {rank}: Pre-broadcast loss {loss.item()}")
                torch.distributed.broadcast(loss, src=last_stage_rank, group=pp_group)
                logger.info(f"Rank {rank}: Post-broadcast loss {loss.item()}")

            eval_losses.append(loss.item())
            eval_perplexities.append(torch.exp(loss).item())
            num_samples += input_ids.size(0)

            if num_samples >= job_config.evaluation.num_samples:
                break

    # Synchronize evaluation results across all ranks
    total_samples = torch.tensor(num_samples, device=device_type)
    torch.distributed.all_reduce(total_samples)

    if total_samples.item() > 0:
        avg_loss = sum(eval_losses) / len(eval_losses)
        avg_perplexity = sum(eval_perplexities) / len(eval_perplexities)
        logger.info(f"Before, evaluation - Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}, num losses {len(eval_losses)}")

        if world_mesh.mesh_dim_names is not None and "dp" in world_mesh.mesh_dim_names:
            avg_loss = torch.tensor(avg_loss, device=device_type)
            avg_perplexity = torch.tensor(avg_perplexity, device=device_type)
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.AVG,
                                         group=world_mesh["dp"].get_group())
            torch.distributed.all_reduce(avg_perplexity, op=torch.distributed.ReduceOp.AVG,
                                         group=world_mesh["dp"].get_group())
            avg_loss = avg_loss.item()
            avg_perplexity = avg_perplexity.item()
        logger.info(f"losses: {eval_losses}")
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}, num losses {len(eval_losses)}")

        return avg_loss, avg_perplexity
    else:
        logger.warning("No samples were evaluated.")
        return None, None


def run_eval():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank == 0:
        print("Hello from rank 0")
        pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    # if rank == 1:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6792, stdoutToServer=True, stderrToServer=True)
    # if rank == 2:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6793, stdoutToServer=True, stderrToServer=True)
    # if rank == 3:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6794, stdoutToServer=True, stderrToServer=True)
    init_logger()
    args = parse_args()
    job_config = setup_job_config(args)

    num_threads = os.cpu_count()  # Set to the number of available CPU cores
    num_threads_per_rank = max(1, num_threads // min(world_size, 8))
    torch.set_num_threads(num_threads_per_rank)
    device = torch.device(f"{device_type}:{rank}")
    torch.cuda.set_device(device)

    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    model = build_reference_model(job_config, tokenizer)

    eval_data_loader = build_hh_data_loader(
        tokenizer,
        job_config.evaluation.batch_size,
        job_config.training.seq_len,
        split="test",
        mode=job_config.training.dataset_mode,
        packing=job_config.training.dataset_packing
    )

    loss_fn = Objective.get_loss_function(job_config.training.loss_function)
    eval_iter = iter(eval_data_loader)

    loss, perplexity = evaluate(model, rank, world_size, eval_iter, loss_fn, model.device_mesh, device_type, job_config)

    if torch.distributed.get_rank() == 0 and loss is not None:
        logger.info(f"Final Evaluation - Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")

    torch.distributed.barrier()
    torch.cuda.synchronize()
    torch.distributed.destroy_process_group()

    sys.exit(0)

def main():
    args = parse_args()
    if args.run_test:
        run_eval()
    else:
        run_torchrun(__file__, args)


if __name__ == "__main__":
    main()