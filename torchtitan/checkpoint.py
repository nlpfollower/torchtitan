# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import enum
import functools
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from multiprocessing import get_context
from typing import Any, Dict, List, Union, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import DefaultSavePlanner, SavePlan, Metadata, WriteItem, FileSystemWriter
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import init_logger, logger
from torchtitan.models.file_reader import OptimizedFileSystemReader
from torchtitan.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.utils import GarbageCollection


class IntervalType(enum.Enum):
    SECONDS = enum.auto()
    STEPS = enum.auto()


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


class ModelWrapper(Stateful):
    def __init__(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self) -> Dict[str, Any]:
        return {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class Terminate:
    pass


class SaveDone:
    pass


@torch.no_grad()
def save_with_gc(state, checkpoint_id, planner=None):
    dcp.save(state, checkpoint_id=checkpoint_id)
    GarbageCollection.collect("GC collection invoked by checkpointer.")


@torch.no_grad()
def save_with_gc_with_custom_writer(state, checkpoint_id, job_config):
    """Save using custom storage writer and planner for unsharded checkpoint."""
    from torchtitan.utils import GarbageCollection
    import torch.distributed.checkpoint as dcp

    is_infra_node0 = getattr(job_config.checkpoint, 'is_infra_node0', False)
    ranks_per_node = getattr(job_config.checkpoint, 'ranks_per_node', 8)

    logger.info(f"Starting custom unsharded save with is_infra_node0={is_infra_node0}")

    # Create the planner with unsharding enabled
    planner = DynamicNode0CheckpointPlanner(
        ranks_per_node=ranks_per_node,
        is_infra_node0=is_infra_node0,
    )

    # Create storage writer
    storage_writer = InfraNode0StorageWriter(
        path=checkpoint_id,
        is_infra_node0=is_infra_node0,
        ranks_per_node=ranks_per_node,
    )

    # Save with custom planner
    metadata = dcp.save(
        state,
        storage_writer=storage_writer,
        planner=planner
    )

    GarbageCollection.collect("GC collection invoked by checkpointer.")

    return metadata


def checkpoint_mp(recv, send):
    init_logger()
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id = obj
            save_with_gc(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish saving the checkpoint in the background process in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


class CheckpointManager:
    def __init__(
            self,
            dataloader: DataLoader,
            model_parts: List[nn.Module],
            optimizers: OptimizersContainer,
            lr_schedulers: LRSchedulersContainer,
            states: Dict[str, Any],
            job_config: JobConfig,
    ) -> None:
        self.job_config = job_config
        ckpt_config = job_config.checkpoint
        self.enable_checkpoint = ckpt_config.enable_checkpoint
        self.keep_latest_k = ckpt_config.keep_latest_k

        # New tensor preload options
        self.use_tensor_preload = ckpt_config.use_tensor_preload
        self.preload_timeout = ckpt_config.preload_timeout
        self.preload_run_id = ckpt_config.preload_run_id or str(uuid.uuid4())

        if not self.enable_checkpoint:
            return
        """
        Note: Pipeline Parallelism and Virtual Stages

        1. even for simple PP schedules, there is a separate optimizer each PP rank.
        rank0's optimizer would have a param_group[0] which refers to layers.0 in the original model.
        rank1's would _also_ have a param_group[0], since it's index based, but referring to layers.1.
        When saving, these collide and one of them is lost.  Then when reloading, only one stage can
        restore its optimizer states, others will error.

            The solution to this problem is optimizer flattening: it landed in #127071 and is enabled in TorchTitan
            by passing the 'flatten_optimizer_state_dict' kwarg to DCP functions called in the OptimizerContainer.

        2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds challenge (1) by also
        requiring us to reason about multiple 'optim' objects locally.

            We solve this in the Model and Optimizer wrapper classes by flattening the state dicts from each object
            into one state dict before saving/loading. We rely on the individual state_dicts to not collide,
            which is gauranteed for the model by correct pipeline splitting and for the optimizer by the flattening
            support described in (1).

        3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers with the assumption that
        all lr_schedulers have the same state_dict.
        """
        self.states = states

        self.states.update(
            {
                "model": ModelWrapper(model_parts),
                "optimizer": optimizers,
                "dataloader": dataloader,
                "lr_scheduler": lr_schedulers,
            }
        )

        self.folder = os.path.join(job_config.job.dump_folder, ckpt_config.folder)
        self.interval_type = (
            IntervalType.SECONDS
            if ckpt_config.interval_type == "seconds"
            else IntervalType.STEPS
        )
        self.interval = ckpt_config.interval
        self.begin_time = 0
        self.time_sync_work = None
        self.time_sync_result = None
        async_mode = ckpt_config.async_mode.lower()
        if async_mode == AsyncMode.ASYNC or self.interval_type == IntervalType.SECONDS:
            self.pg = dist.new_group(backend="gloo")

        self.model_weights_only = ckpt_config.model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[ckpt_config.export_dtype]
        self.exclude_from_loading = ckpt_config.exclude_from_loading

        self.mp = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_id = None
            self.staging_stream = torch.cuda.Stream()
        else:
            raise ValueError(f"Unkown checkpoint async_mode {ckpt_config.async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        if self.enable_checkpoint and self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self) -> None:
        self.begin_time = time.monotonic()

    def _create_checkpoint_id(self, step: int, is_final: bool = False) -> str:
        if is_final:
            return os.path.join(self.folder, "step-final")
        return os.path.join(self.folder, f"step-{step}")

    def _save_last_step(self, curr_step: int, is_final: bool = False) -> None:
        """Fixed version of _save_last_step that properly handles model state dict."""
        if self.model_weights_only:
            # Get the model state dict
            model_state_dict = self.states["model"].state_dict()

            # Create new states dict with "model." prefix preserved
            prefixed_state_dict = {
                f"model.{k}": v for k, v in model_state_dict.items()
            }

            # Keep freqs_cis - don't remove it
            # prefixed_state_dict.pop("model.freqs_cis", None)  # COMMENTED OUT

            # Convert dtype if needed
            if self.export_dtype != torch.float32:
                prefixed_state_dict = {
                    k: v.to(self.export_dtype) if torch.is_tensor(v) else v
                    for k, v in prefixed_state_dict.items()
                }

            # Save the prefixed state dict directly
            states_to_save = prefixed_state_dict

            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} "
                f"at {'final' if is_final else 'last'} step, step {curr_step}."
            )
        else:
            states_to_save = self.states
            logger.info(f"Saving a full checkpoint at {'final' if is_final else 'last'} step, step {curr_step}.")

        # Save with custom writer
        save_with_gc(
            states_to_save,
            checkpoint_id=self._create_checkpoint_id(curr_step, is_final),
        )
        self.reset()

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if not force:
            if self.interval_type == IntervalType.STEPS and not (
                    curr_step % self.interval == 0
            ):
                return False
            if self.interval_type == IntervalType.SECONDS:
                time_sync_result = (time.monotonic() - self.begin_time) >= self.interval
                self.time_sync_result = torch.tensor(int(time_sync_result))
                if self.time_sync_work is None:
                    self.time_sync_work = dist.all_reduce(
                        self.time_sync_result, group=self.pg, async_op=True
                    )
                    return False
                elif curr_step % 5 == 4:
                    self.time_sync_work.wait()
                    self.time_sync_work = None
                    time_sync_result = self.time_sync_result.item()
                    self.time_sync_result = None
                    if time_sync_result == 0:
                        return False
                else:
                    return False

        if self.time_sync_work:
            self.time_sync_work.wait()
            self.time_sync_work = None
            self.time_sync_result = None

        return True

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the background process to finish, {time.monotonic()=}.:.2f"
            )
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()

    def _async_with_pinned_memory(self, checkpoint_id: str) -> None:
        try:
            from torch.distributed._state_dict_utils import (
                _copy_state_dict,
                _create_cpu_state_dict,
            )
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True, share_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_id = checkpoint_id

    def save(self, curr_step: int, force: bool = False, is_final: bool = False) -> None:
        """
        force = True will force the checkpoint to be saved, even if the interval
        has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.
        """
        if not self._should_save(curr_step, force):
            return

        begin = time.monotonic()
        checkpoint_id = self._create_checkpoint_id(curr_step, is_final)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if force:
            self._save_last_step(curr_step, is_final)
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self._async_with_pinned_memory(checkpoint_id)
        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.async_future = dcp.async_save(
                self.states, checkpoint_id=checkpoint_id, process_group=self.pg
            )
        else:
            # Get the planner for regular save
            save_with_gc(
                self.states,
                checkpoint_id=checkpoint_id,
            )
        self.reset()
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled)"
            f"in {time.monotonic() - begin:.2f} seconds."
        )

    def maybe_wait_for_staging(self) -> None:
        if (
                self.enable_checkpoint
                and self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
                and self.staging
        ):
            if not self.staging_stream.query():
                self.staging_stream.synchronize()

            def sync_func():
                self.mp_queue_send.put_nowait(
                    (self.cpu_offload_state_dict, self.staging_id)
                )

            # This may be a faster way to do zero-overhead checkpointing staging
            # checkpointing but we need more thorough investigation before
            # swithing to this method.
            # self.my_thread = threading.Thread(target=func).start()
            sync_func()
            self.staging = False

    def load(self, step: int = -1, checkpoint_path: Optional[str] = None) -> bool:
        if not self.enable_checkpoint:
            return False
        if not os.path.isdir(self.folder):
            return False

        if checkpoint_path is not None:
            checkpoint_id = checkpoint_path
            step = 0
        else:
            if step != -1 and not os.path.isdir(self._create_checkpoint_id(step)):
                return False

            if step == -1:
                step_counts = []
                for filename in os.listdir(self.folder):
                    match = re.search(r"step-(\d+)", filename)
                    metadata_probe = os.path.join(self.folder, filename, ".metadata")
                    if match and os.path.isfile(metadata_probe):
                        step_counts.append(int(match.group(1)))
                if not step_counts:
                    return False
                step = max(step_counts)
            checkpoint_id = self._create_checkpoint_id(step)

        # We won't have optimizer states to load, if we are loading a seed checkpoint
        states = {"model": self.states["model"]} if step == 0 else self.states
        # PyTorch bug: (pytorch/pytorch#138575)
        # dcp.load() replaces the values of stateful elements in `states` with new objects
        # from loading the checkpoint, in addition to updating the states of the original
        # objects from `states` in-place. This is a problem because the state_dict no longer
        # refers to the objects being used in the train loop, meaning any future checkpoints
        # will not include updates to these objects (such as updated optimizer states, etc.)
        original_stateful_states = {
            k: v for k, v in states.items() if isinstance(v, Stateful)
        }
        logger.info(f"Loading the checkpoint at step {step}.")
        begin = time.monotonic()
        states_to_load = {
            k: v for k, v in states.items() if k not in self.exclude_from_loading
        }
        for exclude_key in self.exclude_from_loading:
            if exclude_key not in states:
                raise ValueError(f"{exclude_key} not found in state_dict.")
        if self.use_tensor_preload:
            complete_file = f"/tmp/tensor_preload_{self.preload_run_id}_complete"

            # Wait for file to exist
            max_wait = self.preload_timeout
            wait_time = 0
            while not os.path.exists(complete_file) and (max_wait == 0 or wait_time < max_wait):
                time.sleep(1)
                wait_time += 1
                if wait_time % 10 == 0:
                    logger.info(f"Still waiting for tensor preload (run_id: {self.preload_run_id}) after {wait_time}s")

            dcp.load(
                states_to_load,
                checkpoint_id=checkpoint_id,
                storage_reader=OptimizedFileSystemReader(checkpoint_id, num_threads=16, cuda_streams=16),
            )
        else:
            logger.info("Loading checkpoint without tensor preloading.")
            dcp.load(states_to_load, checkpoint_id=checkpoint_id)
        states.update(states_to_load)
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        # bugfix from above: restore the original stateful objects,
        # whose states were already updated in-place by dcp.load()
        states.update(original_stateful_states)
        GarbageCollection.collect("GC collection for checkpoint loading.")
        return True

    def _purge_stale_checkpoints(self):
        if self.keep_latest_k > 0:
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                if filename == "step-final":
                    continue  # Skip the final checkpoint
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                logger.info(f"Deleting old checkpoint {path}")
                shutil.rmtree(path, ignore_errors=True)


class DynamicNode0CheckpointPlanner(DefaultSavePlanner):
    """
    Custom planner that ensures all data is saved only on infrastructure node 0.
    """

    def __init__(self, *args, **kwargs):
        self.ranks_per_node = kwargs.pop('ranks_per_node', 8)
        self.is_infra_node0 = kwargs.pop('is_infra_node0', False)

        # Keep sharded format - don't flatten/unshard
        super().__init__(
            *args,
            flatten_state_dict=True,
            flatten_sharded_tensors=False,  # Keep tensors sharded
            dedup_replicated_tensors=True,
            **kwargs
        )

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self._determine_save_ranks()

    def _determine_save_ranks(self):
        """Determine which ranks should save based on infrastructure node 0 flag."""
        if not dist.is_initialized():
            self.save_ranks = [0]
            self.should_save = True
            return

        # Create a tensor to gather node 0 information from all ranks
        node0_indicator = torch.tensor(1 if self.is_infra_node0 else 0, dtype=torch.int32).cuda()

        # Gather from all ranks
        all_indicators = [torch.zeros(1, dtype=torch.int32).cuda() for _ in range(self.world_size)]
        dist.all_gather(all_indicators, node0_indicator)

        # Find ALL ranks on infrastructure node 0
        self.save_ranks = []
        for rank, indicator in enumerate(all_indicators):
            if indicator.item() == 1:
                self.save_ranks.append(rank)

        self.should_save = self.rank in self.save_ranks

        logger.info(f"Rank {self.rank}: is_infra_node0={self.is_infra_node0}, "
                    f"save_ranks={self.save_ranks}, should_save={self.should_save}")

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        """
        Create a global plan that only allows infrastructure node 0 ranks to save.
        Each rank keeps its own items to avoid cross-rank data access issues.
        """
        # Let the parent handle the default planning first
        global_plans, metadata = super().create_global_plan(all_plans)

        if not self.save_ranks:
            logger.warning("No save ranks found, using default plan")
            return global_plans, metadata

        # Create redistributed plans
        redistributed_plans = []

        for rank_idx, plan in enumerate(global_plans):
            if rank_idx in self.save_ranks:
                # This rank is on node 0 - keep its items
                redistributed_plans.append(plan)
                logger.info(f"Rank {rank_idx} (on node 0) will write {len(plan.items)} items")
            else:
                # This rank is NOT on node 0 - clear its items
                redistributed_plans.append(SavePlan(
                    items=[],
                    storage_data=plan.storage_data,
                    planner_data=plan.planner_data
                ))

        return redistributed_plans, metadata


class InfraNode0StorageWriter(FileSystemWriter):
    """
    Storage writer that works with DynamicNode0CheckpointPlanner.
    """

    def __init__(self, path: Union[str, os.PathLike], is_infra_node0: bool = False, ranks_per_node: int = 8, **kwargs):
        # Remove our custom args before passing to parent
        super().__init__(path, **kwargs)

        self.is_infra_node0 = is_infra_node0
        self.ranks_per_node = ranks_per_node
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(f"InfraNode0StorageWriter initialized on rank {self.rank}, "
                    f"is_infra_node0={is_infra_node0}")

    def write_data(self, plan: SavePlan, planner: dcp.SavePlanner) -> Any:
        """Override to only write data on infrastructure node 0 ranks."""
        if not plan.items:
            # This rank has nothing to save
            logger.info(f"Rank {self.rank} has no items to save")
            from concurrent.futures import Future
            fut = Future()
            fut.set_result([])
            return fut

        logger.info(f"Rank {self.rank} writing {len(plan.items)} items")
        return super().write_data(plan, planner)