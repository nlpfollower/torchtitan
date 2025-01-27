import os

import torch
from torch.distributed.tensor import DeviceMesh
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims, models_pipelining_fns
from torchtitan.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.utils import get_device_info
from scripts.generate._generation import generate, generate_next_token
import torch.nn as nn

class ReferenceModel(nn.Module):
    def __init__(self, model_parts, pp_schedule=None):
        super().__init__()
        self.model_parts = nn.ModuleList(model_parts)
        self.pp_schedule = pp_schedule

    def forward(self, x):
        if self.pp_schedule:
            return self.pp_schedule(x)
        else:
            for part in self.model_parts:
                x = part(x)
            return x

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, seed=None):
        return generate(self, input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)

    def generate_next_token(self, x, temperature=1.0, top_k=None, rng=None):
        return generate_next_token(self, x, temperature=temperature, top_k=top_k, rng=rng)

def build_reference_model(job_config, world_mesh, parallel_dims, optimizers, lr_schedulers):
    device_type, _ = get_device_info()
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")

    # Create a separate parallel dimension for the reference model
    world_size = job_config.reference_model.data_parallel_shard_degree * job_config.reference_model.tensor_parallel_degree * job_config.reference_model.pipeline_parallel_degree
    reference_parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=job_config.reference_model.data_parallel_shard_degree,
        cp=1,
        tp=job_config.reference_model.tensor_parallel_degree,
        pp=job_config.reference_model.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=False
    )

    # Create a separate device mesh for the reference model
    reference_mesh = reference_parallel_dims.build_mesh(device_type)

    # Load the reference model configuration
    model_cls = model_name_to_cls[job_config.model.name]
    model_config = models_config[job_config.model.name][job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type

    # Build the reference model
    with torch.device("meta"):
        reference_model = model_cls.from_model_args(model_config)
        reference_model.half()
        reference_model.eval()

    # Apply parallelism to the reference model
    if reference_parallel_dims.pp_enabled:
        # Apply pipeline parallelism
        pp_schedule, reference_model_parts = models_pipelining_fns[job_config.model.name](
            reference_model, reference_mesh["pp"], reference_parallel_dims, job_config, device, model_config, None
        )
        for m in reference_model_parts:
            models_parallelize_fns[job_config.model.name](m, reference_mesh, reference_parallel_dims, job_config)
            m.to_empty(device=device)
    else:
        # Apply tensor parallelism
        models_parallelize_fns[job_config.model.name](reference_model, reference_mesh, reference_parallel_dims, job_config)
        reference_model.to_empty(device=device)
        reference_model_parts = [reference_model]

        # Load the checkpoint
    checkpoint = CheckpointManager(
        dataloader=None,
        model_parts=reference_model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"model": ModelWrapper(reference_model_parts)},
        job_config=job_config,
    )
    checkpoint.load(step=0, checkpoint_path=job_config.reference_model.checkpoint_path)

    return ReferenceModel(reference_model_parts, pp_schedule if reference_parallel_dims.pp_enabled else None)