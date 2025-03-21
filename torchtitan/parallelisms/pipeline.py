# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.pipelining._debug import map_debug_info
from torch.distributed.pipelining._utils import flatten_args
from torch.distributed.pipelining.stage import _normalize_model_output_as_tuple, _PipelineStageBase

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    _PipelineScheduleRuntime,
    get_schedule_class,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.distributed.pipelining.stage import PipelineStage

from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger


__all__ = ["build_pipeline_schedule", "generate_split_points", "stage_ids_this_rank"]


# TODO: It's unclear if this API is general enough to be used by other models.
# If not, we should move it to a Transformer-specific directory.
def generate_split_points(
    job_config: JobConfig, pp_dim: int, num_layers: int
) -> list[str]:
    """
    Generate a default split point based on the number of layers and
    pipeline parallel dimension.

    Args:
        job_config (JobConfig): The job configuration.
        pp_dim (int): The pipeline parallel dimension.
        num_layers (int): The number of layers in the model.

    Returns:
        list[str]: A list of split point FQNs.
    """

    schedule_class = get_schedule_class(
        job_config.experimental.pipeline_parallel_schedule
    )
    if issubclass(schedule_class, PipelineScheduleSingle):
        num_stages_per_rank = 1
    elif issubclass(schedule_class, PipelineScheduleMulti):
        # Multi-stage schedules support more than 2 stages per rank, but this is the default if
        # no pipeline split is specified
        num_stages_per_rank = 2
    else:
        raise ValueError(
            f"Unsupported pipeline schedule: {job_config.experimental.pipeline_parallel_schedule}"
        )
    total_stages = pp_dim * num_stages_per_rank
    if total_stages > num_layers:
        raise ValueError("Total stages cannot be greater than the number of layers")

    base_interval = num_layers // total_stages
    extra_layers = num_layers % total_stages

    splits = []
    current_layer = 0
    for i in range(total_stages - 1):
        if i == 0:
            current_layer += base_interval
        else:
            # Middle stages get an extra layer if there are any remaining
            if extra_layers > 0:
                current_layer += base_interval + 1
                extra_layers -= 1
            else:
                current_layer += base_interval
        splits.append("layers." + str(current_layer))
    logger.info(
        f"No 'pipeline_parallel_split_points' provided so the generated splits are: {splits} "
        "This may be sub-optimal as the number of layers per stage may be unbalanced."
    )
    return splits


def build_pipeline_schedule(
    job_config: JobConfig, stages: list[PipelineStage], loss_fn: Callable
) -> _PipelineSchedule:
    """Builds a pipeline schedule for the given job configuration and stages.

    Args:
        job_config (JobConfig): The job configuration.
        stages (list[PipelineStage]): The stages to be scheduled.
        loss_fn (Callable): The loss function.

    Returns:
        _PipelineSchedule: The pipeline schedule for the given stages.
    """
    pp_schedule_csv = job_config.experimental.pipeline_parallel_schedule_csv

    # Validate that pp_schedule_csv is a valid path
    if pp_schedule_csv:
        if not os.path.isfile(pp_schedule_csv):
            raise FileNotFoundError(
                f"The specified path {pp_schedule_csv} does not exist or is not a file."
            )
        schedule_class = _PipelineScheduleRuntime
    else:
        schedule_class = get_schedule_class(
            job_config.experimental.pipeline_parallel_schedule
        )

    looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
    n_microbatches = job_config.experimental.pipeline_parallel_microbatches
    # We expect that the number of local stages (`len(stages)`) is the same across all ranks
    num_total_stages = job_config.experimental.pipeline_parallel_degree * len(stages)
    if n_microbatches is None:
        n_microbatches = num_total_stages
    elif n_microbatches < num_total_stages:
        logger.warning(
            f"Number of microbatches ({n_microbatches}) is less than the total number "
            f"of stages ({num_total_stages}) which may result in a bubble in the pipeline."
        )

    # validate that the batch size is divisible by the number of microbatches otherwise we'll hang or error during training
    if job_config.training.batch_size % n_microbatches != 0:
        raise ValueError(
            f"Batch size {job_config.training.batch_size} must be divisible by number of microbatches {n_microbatches}. "
            "Update the config arguments for either batch_size or pipeline_parallel_microbatches."
        )

    schedule = schedule_class(
        stages if looped_schedule else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
    )
    logger.info(
        f"Using pipeline schedule {job_config.experimental.pipeline_parallel_schedule} "
        f"with {n_microbatches} microbatches and {num_total_stages} stages."
    )

    if pp_schedule_csv:
        assert schedule_class in [
            PipelineScheduleSingle,
            PipelineScheduleMulti,
            _PipelineScheduleRuntime,
        ], (
            "Only PipelineScheduleSingle (single stage), PipelineScheduleMulti (multistage), "
            "and _PipelineScheduleRuntime support csv schedules"
        )
        schedule._load_csv(pp_schedule_csv)

    return schedule


# TODO(whc) should this be a utility inside torch.pipelining?
def stage_ids_this_rank(
    pp_rank: int, pp_size: int, num_stages: int, style: str = "loop"
) -> tuple[int]:
    """Compute the stage ids for the stages that will run on this pp rank for either a looped or V style schedule"""
    assert (
        num_stages % pp_size == 0
    ), f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size}"
    stages_per_rank = num_stages // pp_size
    if style == "loop":
        return tuple(pp_rank + s * pp_size for s in range(stages_per_rank))
    elif style == "v":
        assert (
            stages_per_rank == 2
        ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
        stage_v_pairs = list(
            zip(range(pp_size), range(num_stages - 1, pp_size - 1, -1))
        )
        return stage_v_pairs[pp_rank]


def pipeline_forward(
        stages: List[Any],
        pp_size: int,
        inputs: Any,
        mask: Optional[torch.Tensor] = None,
        stages_initialized: bool = True,
) -> Tuple[Any, bool]:
    """
    Performs a forward-only pass through pipeline stages in the correct sequence.

    Args:
        stages: List of pipeline stages available on this rank
        pp_size: Pipeline parallel size
        inputs: Input tensor(s) to process
        mask: Optional attention mask
        stages_initialized: Whether stages have been initialized already

    Returns:
        tuple: (output tensor or None, has_last_stage flag)
    """
    if not stages:
        return None, False

    # Setup state information
    mb_args = (inputs,) if not isinstance(inputs, tuple) else inputs
    stage_map = {stage.stage_index: stage for stage in stages}
    has_last_stage = any(stage.is_last for stage in stages)
    total_stages = len(stage_map) * pp_size if pp_size else max(stage_map.keys()) + 1

    # Initialize stages if needed
    next_args: tuple[Any, ...] = tuple()
    if not stages_initialized:
        for stage in sorted(stages, key=lambda s: s.stage_index):
            if stage.stage_index == 0:
                next_args = stage._prepare_forward_infra(1, mb_args, {})
            else:
                next_args = stage._prepare_forward_infra(1, next_args, {})

    # Process all stages in sequence
    ops = []
    output = None

    # Clear runtime states for all stages on this rank
    for stage in stages:
        stage.clear_runtime_states()

    # Process through all pipeline stages
    for stage_idx in range(total_stages):
        current_stage = stage_map.get(stage_idx)

        if current_stage:
            if stage_idx > 0 and stage_idx - 1 not in stage_map:
                # We need to receive from the previous stage
                ops.extend(current_stage.get_fwd_recv_ops(0))
                if ops:
                    _batch_p2p(ops).wait()
                    ops = []

            # Process this stage
            mb_kwargs = {"mask": mask} if mask is not None else {}
            output = current_stage.forward_one_chunk(0, mb_args, mb_kwargs)
            mb_args = (output,)

            if stage_idx < total_stages - 1:
                # We need to send to the next stage
                ops.extend(current_stage.get_fwd_send_ops(0))
                next_stage = stage_map.get(stage_idx + 1)
                if next_stage:
                    ops.extend(next_stage.get_fwd_recv_ops(0))
                if ops:
                    _batch_p2p(ops).wait()
                    ops = []
            current_stage.clear_runtime_states()

    return output, has_last_stage


def _batch_p2p(p2p_ops: list[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed
    """
    if len(p2p_ops) == 0:
        return None
    return dist.batch_isend_irecv(p2p_ops).pop()


class PipelineEphemeralContext:
    """
    Context manager for temporarily working with pipeline stages in a clean state.

    Creates an ephemeral execution environment for pipeline operations (like evaluation) by completely resetting pipeline
    stages to their initial state, then restoring the original training state upon exit.

    This context allows evaluation to run with different batch sizes or configurations without affecting the persistent
    state needed for training. All stage metadata, communication buffers, and autograd state are preserved.

    Usage:
        with PipelineEphemeralContext(stages):
            # Perform evaluation or other operations requiring fresh stage state
            # Original training state is automatically restored when exiting
    """

    def __init__(self, stages):
        """Initialize with pipeline stages."""
        self.stages = stages
        self.saved_state = {}

    def __enter__(self):
        """Save critical state from all pipeline stages and reset to fresh state."""
        if not self.stages:
            return self

        # Save and reset each stage
        for i, stage in enumerate(self.stages):
            # Create a nested dictionary for this stage
            stage_state = {}

            # Save the output metadata (critical for tensor shape validation)
            stage_state['_outputs_meta'] = stage._outputs_meta

            # Save communication infrastructure
            stage_state['args_recv_info'] = dict(stage.args_recv_info) if hasattr(stage, 'args_recv_info') else {}
            stage_state['act_send_info'] = dict(stage.act_send_info) if hasattr(stage, 'act_send_info') else {}
            stage_state['grad_recv_info'] = dict(stage.grad_recv_info) if hasattr(stage, 'grad_recv_info') else {}
            stage_state['grad_send_info'] = stage.grad_send_info

            # Save chunks configuration
            stage_state['chunks'] = stage.chunks

            # Save input shape metadata if present
            if hasattr(stage, 'inputs_meta'):
                stage_state['inputs_meta'] = stage.inputs_meta

            # Track whether stages were initialized
            if hasattr(stage, '_stage_initialized'):
                stage_state['_stage_initialized'] = stage._stage_initialized

            # Save backward state
            stage_state['backward_state'] = dict(stage.backward_state)
            stage_state['dw_runner'] = dict(stage.dw_runner)
            stage_state['has_backward'] = stage.has_backward

            # Store all saved state for this stage
            self.saved_state[i] = stage_state

            # Reset to initial state (as if freshly created)
            stage._outputs_meta = None
            stage.args_recv_info = {}
            stage.act_send_info = {}
            stage.grad_recv_info = {}
            stage.grad_send_info = None
            stage.chunks = None
            if hasattr(stage, 'inputs_meta'):
                stage.inputs_meta = None
            if hasattr(stage, '_stage_initialized'):
                stage._stage_initialized = False
            stage.backward_state = {}
            stage.dw_runner = {}

            # Clear runtime states
            stage.clear_runtime_states()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore saved state to all pipeline stages."""
        if not self.stages or not self.saved_state:
            return

        for i, stage in enumerate(self.stages):
            if i not in self.saved_state:
                continue

            stage_state = self.saved_state[i]

            # Restore critical shape validation metadata
            stage._outputs_meta = stage_state['_outputs_meta']

            # Restore communication infrastructure
            stage.args_recv_info = stage_state['args_recv_info']
            stage.act_send_info = stage_state['act_send_info']
            stage.grad_recv_info = stage_state['grad_recv_info']
            stage.grad_send_info = stage_state['grad_send_info']

            # Restore chunks configuration
            stage.chunks = stage_state['chunks']

            # Restore input shape metadata if it was saved
            if 'inputs_meta' in stage_state and hasattr(stage, 'inputs_meta'):
                stage.inputs_meta = stage_state['inputs_meta']

            # Restore initialization flag
            if '_stage_initialized' in stage_state and hasattr(stage, '_stage_initialized'):
                stage._stage_initialized = stage_state['_stage_initialized']

            # Restore backward state
            stage.backward_state = stage_state['backward_state']
            stage.dw_runner = stage_state['dw_runner']
            stage.has_backward = stage_state['has_backward']

            # Clear runtime states to start fresh
            stage.clear_runtime_states()

def monkey_patch_pipeline_schedule():
    """
    Monkey-patches the PipelineSchedule to properly handle loss computation
    with additional kwargs specific to each microbatch.
    """
    from torch.distributed.pipelining.schedules import _PipelineSchedule
    import logging

    logger = logging.getLogger(__name__)

    # Store original methods
    original_maybe_compute_loss = _PipelineSchedule._maybe_compute_loss
    original_compute_loss = _PipelineSchedule._compute_loss

    def patched_maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        """
        Patched version that passes mb_index to _compute_loss.
        """
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index], mb_index)
            self._internal_losses.append(loss)
            return loss
        return None

    def patched_compute_loss(self, output, target, mb_index):
        """
        Patched version that passes mb_index to the loss function.
        """
        return self._loss_fn(output, target, mb_index=mb_index)

    # Apply the patches
    _PipelineSchedule._maybe_compute_loss = patched_maybe_compute_loss
    _PipelineSchedule._compute_loss = patched_compute_loss

    # logger.info("Successfully monkey-patched PipelineSchedule loss computation methods")

    # Return a function to restore the original if needed
    def restore_original():
        _PipelineSchedule._maybe_compute_loss = original_maybe_compute_loss
        _PipelineSchedule._compute_loss = original_compute_loss
        # logger.info("Restored original PipelineSchedule methods")

    return restore_original


def monkey_patch_pipeline_stage():
    """
    Monkey-patches PyTorch's pipeline stage implementation to exclude specified kwargs
    from backward pass computation.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Store original method for reference and safety
    original_forward_one_chunk = _PipelineStageBase.forward_one_chunk

    def patched_forward_one_chunk(self, fwd_chunk_id, args, kwargs=None):
        """
        Patched version of forward_one_chunk that excludes specified kwargs from backward computation.
        """
        if self.is_first:
            composite_args = args
        else:
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

        # Extract mask from kwargs for forward pass, but don't include in backward
        mask = composite_kwargs.pop('mask', None) if composite_kwargs else None
        forward_kwargs = composite_kwargs.copy()
        if mask is not None:
            forward_kwargs['mask'] = mask

        self._validate_fwd_input(args, forward_kwargs)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **forward_kwargs)

        except Exception as e:
            exc_msg = f"""
                {self.log_prefix} failed to run forward:
                args: {map_debug_info(composite_args)}
                kwargs: {map_debug_info(composite_kwargs)}
                """
            raise RuntimeError(exc_msg) from e

        # Normalize output for pipeline
        output_tuple = _normalize_model_output_as_tuple(output)
        self.output_chunks.append(output)

        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )

        self._validate_fwd_outputs(output_tuple)
        return output

    # Apply the patch directly to the class - this is the key fix
    _PipelineStageBase.forward_one_chunk = patched_forward_one_chunk
    # logger.info("Successfully monkey-patched _PipelineStageBase.forward_one_chunk")

    # Return a function to restore the original if needed
    def restore_original():
        _PipelineStageBase.forward_one_chunk = original_forward_one_chunk
        # logger.info("Restored original _PipelineStageBase.forward_one_chunk")

    return restore_original