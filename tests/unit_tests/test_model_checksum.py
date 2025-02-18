import argparse
import torch
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.checksum import checksum_model
from torchtitan.utils import get_device_info
from torchtitan.parallelisms import ParallelDims
from reference_model import build_reference_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test model checksum with parallelism")
    parser.add_argument("--dp_shard", type=int, default=1)
    parser.add_argument("--dp_replicate", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    init_logger()

    device_type, _ = get_device_info()

    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "3B"
    job_config.model.tokenizer_path = "models/Llama3.2-3B-Instruct/tokenizer.model"
    job_config.training.seq_len = 8192
    job_config.reference_model.checkpoint_path = args.checkpoint_path
    job_config.reference_model.data_parallel_shard_degree = args.dp_shard
    job_config.reference_model.tensor_parallel_degree = args.tp
    job_config.reference_model.pipeline_parallel_degree = args.pp

    world_size = args.dp_shard * args.dp_replicate * args.cp * args.tp * args.pp
    parallel_dims = ParallelDims(
        dp_shard=args.dp_shard,
        dp_replicate=args.dp_replicate,
        cp=args.cp,
        tp=args.tp,
        pp=args.pp,
        world_size=world_size,
        enable_loss_parallel=True
    )

    world_mesh = parallel_dims.build_mesh(device_type)

    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)

    model = build_reference_model(job_config, tokenizer)

    total_checksum, param_checksums = checksum_model(model, world_mesh)

    if torch.distributed.get_rank() == 0:
        logger.info(f"Total model checksum: {total_checksum}")
        logger.info("Parameter checksums:")
        for name, checksum in param_checksums.items():
            logger.info(f"{name}: {checksum}")


if __name__ == "__main__":
    main()