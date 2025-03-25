import os
from typing import Optional, Any

import torch
from torch.distributed.tensor import DeviceMesh
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchdata.nodes import Stateful

from torchtitan import state
from torchtitan.models import model_name_to_cls, models_config
from torchtitan.models.llama import pipeline_llama, parallelize_llama
from torchtitan.parallelisms import ParallelDims
from torchtitan.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.parallelisms.pipeline import pipeline_forward
from torchtitan.utils import get_device_info, set_determinism, create_context_parallel_ctx
from scripts.generate._generation import generate, generate_next_token
import torch.nn as nn
import torch.distributed as dist

class ReferenceModel(nn.Module):
    def __init__(self, model_config, model_parts, device_mesh, stages=None, pp_rank=None, pp_size=None):
        super().__init__()
        self.model_config = model_config
        self.model_parts = nn.ModuleList(model_parts)
        self.device_mesh = device_mesh
        self.stages = stages
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.cp_enabled = "cp" in device_mesh

        # Store CP-specific information if enabled
        if self.cp_enabled:
            self.cp_mesh = device_mesh["cp"]
            self.cp_rotate_method = None  # Will be set during forward calls
        else:
            self.cp_mesh = None

        # Get pipeline order if applicable
        if stages and len(stages) > 0:
            # For ZBV we get 2 stages per rank
            self.total_stages = len(stages) * pp_size
            # Map stage indices to actual stage objects for this rank
            self.stage_map = {stage.stage_index: stage for stage in stages}
            self.has_last_stage = any(stage.is_last for stage in stages)
        else:
            self.has_last_stage = True

        self._stages_initialized = False

    def forward(self, x, mask=None):
        # Setup context parallel context if CP is enabled
        if self.cp_enabled:
            cp_ctx = self._create_cp_context(x, mask)
        else:
            cp_ctx = None

        # Use the CP context for forward pass
        with cp_ctx:
            if self.stages:
                return self._pipeline_forward(x, mask)
            else:
                for part in self.model_parts:
                    x = part(x, mask=mask)
                return x

    def _create_cp_context(self, input_ids, attention_mask=None):
        """Create context parallel context for reference model forward pass."""
        # Initialize buffers, sequence dimensions, and no-restore buffers
        cp_buffers = [input_ids]
        cp_seq_dims = [1]
        cp_no_restore_buffers = {input_ids}

        # Only add attention_mask if it is not None
        if attention_mask is not None:
            cp_buffers.append(attention_mask)
            cp_seq_dims.append(2)
            cp_no_restore_buffers.add(attention_mask)

        # Append the freqs_cis from each model part
        cp_buffers.extend([m.freqs_cis for m in self.model_parts])
        cp_seq_dims.extend([0 for _ in self.model_parts])

        # Create CP context - use the same rotation method as the main model
        return create_context_parallel_ctx(
            cp_mesh=self.cp_mesh,
            cp_buffers=cp_buffers,
            cp_seq_dims=cp_seq_dims,
            cp_no_restore_buffers=cp_no_restore_buffers,
            cp_rotate_method=self.cp_rotate_method,
        )

    def _pipeline_forward(self, x, mask=None):
        """Execute forward pass through pipeline stages in correct sequence."""
        output, has_last_stage = pipeline_forward(
            stages=self.stages,
            pp_size=self.pp_size,
            inputs=x,
            mask=mask,
            stages_initialized=self._stages_initialized,
        )
        self._stages_initialized = True
        return output if has_last_stage else None

    def set_cp_rotate_method(self, method):
        """Set the CP rotation method to ensure consistency with main model."""
        self.cp_rotate_method = method

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, seed=None):
        return generate(self, input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)

    def generate_next_token(self, x, temperature=1.0, top_k=None, rng=None):
        return generate_next_token(self, x, temperature=temperature, top_k=top_k, rng=rng)

def build_reference_model(job_config, tokenizer):
    device_type, _ = get_device_info()
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")

    # Check if context parallel is enabled
    cp_enabled = job_config.experimental.context_parallel_degree > 1

    # Create a separate parallel dimension for the reference model
    world_size = (job_config.reference_model.data_parallel_shard_degree *
                  job_config.reference_model.tensor_parallel_degree *
                  job_config.reference_model.pipeline_parallel_degree *
                  job_config.experimental.context_parallel_degree)
    reference_parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=job_config.reference_model.data_parallel_shard_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.reference_model.tensor_parallel_degree,
        pp=job_config.reference_model.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=False
    )

    # Create a separate device mesh for the reference model
    reference_mesh = reference_parallel_dims.build_mesh(device_type)
    pp_mesh = reference_mesh["pp"] if reference_parallel_dims.pp_enabled else None

    if job_config.training.deterministic:
        set_determinism(
            reference_mesh, device, job_config.training.seed, job_config.training.deterministic
        )

    # Load the reference model configuration
    model_cls = model_name_to_cls[job_config.model.name]
    model_config = models_config[job_config.model.name][job_config.model.flavor]
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    # Build the reference model
    with torch.device("meta"):
        reference_model = model_cls.from_model_args(model_config)
        reference_model.half()
        reference_model.eval()

    # Apply parallelism to the reference model
    stages = None
    if reference_parallel_dims.pp_enabled:
        # Apply pipeline parallelism
        pp_schedule, stages, reference_model_parts, _, _ = pipeline_llama(
            reference_model, reference_mesh["pp"], reference_parallel_dims, job_config, device, model_config, None
        )
        for m in reference_model_parts:
            parallelize_llama(m, reference_mesh, reference_parallel_dims, job_config)
            m.to_empty(device=device)
    else:
        # Apply tensor parallelism
        parallelize_llama(reference_model, reference_mesh, reference_parallel_dims, job_config)
        reference_model.to_empty(device=device)
        reference_model_parts = [reference_model]

    optimizers = MinimalOptimizersContainer(reference_model)
    lr_schedulers = MinimalSchedulersContainer(optimizers.optimizers[0])

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

    # Setup CP info in state if enabled
    # Note: don't setup cp state info, since it's shared across training and reference models

    pp_rank = pp_mesh.get_local_rank() if pp_mesh else None
    pp_size = pp_mesh.size() if pp_mesh else None

    # Create the reference model with CP flag
    ref_model = ReferenceModel(
        model_config,
        reference_model_parts,
        reference_mesh,
        stages=stages,
        pp_rank=pp_rank,
        pp_size=pp_size
    )

    # Set the CP rotation method if CP is enabled
    if cp_enabled:
        ref_model.set_cp_rotate_method(job_config.experimental.context_parallel_rotate_method)

    return ref_model

class MinimalOptimizersContainer(Stateful):
    def __init__(self, model):
        self.optimizers = [Adam(model.parameters())]

    def state_dict(self):
        return {"optimizer_0": self.optimizers[0].state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizers[0].load_state_dict(state_dict["optimizer_0"])

class MinimalSchedulersContainer:
    def __init__(self, optimizer):
        self.schedulers = [LambdaLR(optimizer, lr_lambda=lambda _: 1)]

    def get_lr_scheduler_state(self):
        return {"lr_scheduler": self.schedulers[0]}