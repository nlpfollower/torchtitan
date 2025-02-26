#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import pydevd_pycharm
from tests.unit_tests.test_utils import run_torchrun
from torchtitan.logging import init_logger, logger

import torch
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.checksum import checksum_model
from torchtitan.utils import get_device_info
from torchtitan.parallelisms import ParallelDims
from torchtitan.models.reference_model import build_reference_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test model checksum with parallelism")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--dp_shard", type=int, default=1)
    parser.add_argument("--dp_replicate", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--run_test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()

def run_checksum():
    # pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    args = parse_args()
    init_logger()

    device_type, _ = get_device_info()

    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "3B"
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

    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    model = build_reference_model(job_config, tokenizer)

    total_checksum, param_checksums = checksum_model(model, model.device_mesh)

    if torch.distributed.get_rank() == 0:
        logger.info(f"Total model checksum: {total_checksum}")

    torch.distributed.destroy_process_group()

def main():
    args = parse_args()
    if args.run_test:
        run_checksum()
    else:
        run_torchrun(__file__, args)


if __name__ == "__main__":
    main()