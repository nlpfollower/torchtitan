# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Callable, List, Dict, Any, Optional, Tuple, Union

import torch
import torch.distributed as dist

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
        document_ids: Optional[torch.Tensor] = None,
        stages_initialized: bool = True,
) -> Tuple[Any, bool]:
    """
    Performs a forward-only pass through pipeline stages in the correct sequence.

    Args:
        stages: List of pipeline stages available on this rank
        inputs: Input tensor(s) to process
        mask: Optional attention mask
        document_ids: Optional document IDs for block attention
        stages_initialized: Whether stages have been initialized already
        device_type: Device type (cuda/cpu)
        pp_size: Pipeline parallel size
        pp_rank: Pipeline parallel rank

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

    # Set document_ids in state if available
    if document_ids is not None:
        from torchtitan import state
        state.DOCUMENT_IDS = document_ids

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