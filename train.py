# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import os
import time
from datetime import timedelta

import pydevd_pycharm
import torch
from lxml.html.diff import token

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.experimental._attention import context_parallel_unshard

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.checksum import combine_checksums, finite_field_add, checksum_model
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_custom_data_loader
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.models.llama import pipeline_llama, parallelize_llama
from torchtitan.models.llama.attention_utils import create_block_document_causal_mask
from torchtitan.models.reference_model import build_reference_model
from torchtitan.objective import Objective
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.model_converter import build_model_converters
from torchtitan.parallelisms import ParallelDims
from torchtitan.parallelisms.context import shard_attention_mask
from torchtitan.parallelisms.pipeline import pipeline_forward, \
    monkey_patch_pipeline_schedule, PipelineEphemeralContext
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan import state


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # Apply the pipeline schedule and stage monkey patches before any pipeline initialization
    restore_fn_pipeline_schedule = monkey_patch_pipeline_schedule()

    local_rank = int(os.environ.get("LOCAL_RANK"))
    rank = int(os.environ.get("RANK"))
    logger.info(f"rank {rank} and local rank {local_rank}")
    # if rank == 0:
    #     print("Hello from rank 0")
    #     pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    # if rank == 1:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6792, stdoutToServer=True, stderrToServer=True)
    # if rank == 2:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6793, stdoutToServer=True, stderrToServer=True)
    # if rank == 3:
    #     print("Hello from rank 1")
    #     pydevd_pycharm.settrace('localhost', port=6794, stdoutToServer=True, stderrToServer=True)

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)
    logger.info(f"Torch previous num threads: {torch.get_num_threads()}")
    num_threads = os.cpu_count()  # Set to the number of available CPU cores
    num_threads_per_rank = max(1, num_threads // min(world_size, 8))
    torch.set_num_threads(num_threads_per_rank)
    logger.info(f"Torch new num threads: {torch.get_num_threads()}")

    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
    eval_data_loader = None
    # build dataloader
    if job_config.training.dataset_type == "huggingface":
        data_loader = build_hf_data_loader(
            dp_degree,
            dp_rank,
            tokenizer,
            job_config
        )
    elif job_config.training.dataset_type == "custom":
        data_loader = build_custom_data_loader(
            job_config.training.dataset_path,
            job_config.training.dataset,
            "train",  # or use a config option for split
            tokenizer,
            job_config.training.batch_size,
            job_config.training.seq_len,
            dp_degree,
            dp_rank,
            device_type,
            mode=job_config.training.dataset_mode,
            packing=job_config.training.dataset_packing,
            system_prompt=job_config.training.system_prompt
        )
        if job_config.evaluation.enabled:
            eval_data_loader = build_custom_data_loader(
                job_config.training.dataset_path,
                job_config.training.dataset,
                "test",  # Use a separate split for evaluation
                tokenizer,
                job_config.evaluation.batch_size,
                job_config.training.seq_len,
                dp_degree,
                dp_rank,
                device_type,
                mode=job_config.training.dataset_mode,
                packing=job_config.training.dataset_packing,
                system_prompt=job_config.training.system_prompt
            )
    else:
        raise ValueError(f"Unsupported dataset type: {job_config.training.dataset_type}")

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    # with set_default_dtype(torch.bfloat16), torch.device("meta"):
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # log model size
    model_param_count = utils.get_num_params(model)
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    loss_fn = Objective.get_loss_function(job_config.training.loss_function)

    # TODO: compiling loss function causes CUDA errors, turning off for now
    # if job_config.training.compile:
    #     loss_fn = torch.compile(loss_fn)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            stages,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = pipeline_llama(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            parallelize_llama(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.init_weights(buffer_device=buffer_device)
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        parallelize_llama(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()

        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()
    if parallel_dims.cp_enabled:
        cp_mesh = world_mesh["cp"]
        state.set_cp_info(
            cp_rank=cp_mesh.get_local_rank(),
            cp_world_size=cp_mesh.size(),
            cp_group=cp_mesh.get_group(),
            cp_mesh=cp_mesh
        )

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # if not parallel_dims.pp_enabled:
    #     real_checksum, _ = checksum_model(model, world_mesh)
    #     logger.info(f"Start Checkpoint checksum: {real_checksum}")
    if job_config.reference_model.enabled:
        reference_model = build_reference_model(job_config, tokenizer)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    eval_components = {
        'model_parts': model_parts,
        'reference_model': reference_model if job_config.reference_model.enabled else None,
        'eval_data_loader': eval_data_loader if job_config.evaluation.enabled else None,
        'parallel_dims': parallel_dims,
        'stages': stages if parallel_dims.pp_enabled else None,
        'loss_fn': loss_fn,
        'world_size': world_size,
        'rank': dp_rank,
        'device_type': device_type,
        'world_mesh': world_mesh,
    }
    stages_initialized = False
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        is_dataset_exhausted = torch.zeros(world_size, dtype=torch.bool, device=device)
        model_timers = []  # Will store (forward_time, backward_time) or (step_time) tuples
        eval_timers = []  # Will store eval_time list
        last_log_time = time.perf_counter()
        while train_state.step < job_config.training.steps:
            # Evaluation step
            if job_config.evaluation.enabled and train_state.step % job_config.evaluation.interval == 0:
                eval_start = time.perf_counter()
                evaluate(eval_components, job_config, train_state.step, metric_logger, stages_initialized)
                eval_timers.append(time.perf_counter()-eval_start)
            stages_initialized = False

            train_state.step += 1
            gc_handler.run(train_state.step)

            try:
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                if batch == "end":
                    is_dataset_exhausted[dp_rank] = True
                torch.distributed.all_reduce(is_dataset_exhausted)
                if torch.any(is_dataset_exhausted):
                    logger.info(f"Rank {rank}: All ranks have exhausted their data. Ending training.")
                    break

            except StopIteration:
                logger.warning("DataLoader has exhausted its data. Ending training.")
                break

            try:
                input_ids, labels = batch['input_ids'], batch['labels']
            except TypeError:
                input_ids, labels = batch[0], batch[1]
            document_ids = None
            attention_mask = None
            if 'document_ids' in batch:
                document_ids = batch['document_ids'].to(device_type)
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask'].to(device_type)

            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            optimizers.zero_grad()

            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            if parallel_dims.cp_enabled:
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
                cp_buffers.extend([m.freqs_cis for m in model_parts])
                cp_seq_dims.extend([0 for _ in model_parts])

                optional_context_parallel_ctx = utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=cp_buffers,
                    cp_seq_dims=cp_seq_dims,
                    cp_no_restore_buffers=cp_no_restore_buffers,
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
            else:
                optional_context_parallel_ctx = None

            # Set state tensors (important for both PP and non-PP paths)
            state.set_state_tensors(
                document_ids=document_ids,
                attention_mask=attention_mask,
                batch_size=job_config.training.batch_size,
                n_microbatches=job_config.experimental.pipeline_parallel_microbatches
            )

            reference_logits = None
            if job_config.reference_model.enabled:
                with torch.no_grad():
                    reference_logits = reference_model(input_ids, attention_mask)
                    state.set_state_tensors(reference_logits=reference_logits)

            if parallel_dims.pp_enabled:
                # For PP, measure the entire step time
                pp_step_start = time.perf_counter()

                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)

                    # Pass mask as a regular keyword argument - it will be automatically
                    # split into microbatches by PyTorch's pipeline implementation
                    if has_first_stage:
                        pp_schedule.step(input_ids, target=targets, losses=losses, mask=attention_mask)
                    else:
                        pp_schedule.step(target=targets, losses=losses, mask=attention_mask)

                model_timers.append(time.perf_counter() - pp_step_start)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    forward_start = time.perf_counter()
                    logits = model(input_ids, mask=attention_mask)
                    forward_time = time.perf_counter() - forward_start

                    loss = loss_fn(logits, labels, reference_logits, document_ids)
                    # Free memory before backward pass
                    del logits

                    backward_start = time.perf_counter()
                    loss.backward()
                    backward_time = time.perf_counter() - backward_start
                    model_timers.append((forward_time, backward_time))

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                current_time = time.perf_counter()
                eval_time = sum(eval_timers)

                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(loss, world_mesh["dp_cp"]),
                        utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = current_time - last_log_time - eval_time

                if parallel_dims.pp_enabled:
                    # For PP runs, calculate average step time
                    avg_step_time = sum(model_timers) / len(model_timers) if model_timers else 0
                    timing_log = f"{color.cyan}step: {avg_step_time:.4f}s  "

                    # Add to metrics
                    metrics_timing = {
                        "time_metrics/pp_step(s)": avg_step_time,
                    }
                else:
                    # For non-PP runs, calculate average forward and backward times
                    avg_forward_time = sum(t[0] for t in model_timers) / len(model_timers) if model_timers else 0
                    avg_backward_time = sum(t[1] for t in model_timers) / len(model_timers) if model_timers else 0
                    timing_log = f"{color.cyan}fwd: {avg_forward_time:.4f}s bwd: {avg_backward_time:.4f}s  "

                    # Add to metrics
                    metrics_timing = {
                        "time_metrics/forward(s)": avg_forward_time,
                        "time_metrics/backward(s)": avg_backward_time,
                    }

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * tps / gpu_peak_flops
                tflops = num_flop_per_token * tps / 1e12

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    "tflops": tflops,
                    "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metrics.update(metrics_timing)

                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.red}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.green}local loss: {loss.item():7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,}  "
                    f"{timing_log}"
                    f"{color.cyan}tflops: {tflops:,.2f}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                # Reset all counters after logging
                ntokens_since_last_log = 0
                data_loading_times = []
                model_timers = []
                eval_timers = []
                device_memory_monitor.reset_peak_stats()
                last_log_time = time.perf_counter()

            # save checkpoint. If train step reaches upper bound, skip because it will be saved at the end
            if train_state.step < job_config.training.steps:
                checkpoint.save(train_state.step)

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

        torch.distributed.barrier()
        if torch.any(is_dataset_exhausted) or train_state.step >= job_config.training.steps:
            logger.info("End of training reached.")

            # Run final evaluation if enabled
            if job_config.evaluation.enabled:
                logger.info("Running final evaluation...")
                final_loss, final_perplexity = evaluate(eval_components, job_config, train_state.step, metric_logger, stages_initialized)
                logger.info(f"Final evaluation results - Loss: {final_loss:.4f}, Perplexity: {final_perplexity:.4f}")

            # Save final checkpoint
            logger.info("Saving final checkpoint...")
            checkpoint.save(train_state.step, force=True, is_final=True)

            # if not parallel_dims.pp_enabled:
            #     real_checksum, _ = checksum_model(model, world_mesh)
            #     logger.info(f"Checkpoint - Checkpoint checksum: {real_checksum}")



    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


def evaluate(eval_components, job_config, current_step, metric_logger, stages_initialized):
    logger.info(f"Starting evaluation at step {current_step} with {job_config.evaluation.num_samples} samples")

    model_parts = eval_components['model_parts']
    reference_model = eval_components['reference_model']
    eval_data_loader = eval_components['eval_data_loader']
    parallel_dims = eval_components['parallel_dims']
    stages = eval_components['stages']
    loss_fn = eval_components['loss_fn']
    rank = eval_components['rank']
    world_size = eval_components['world_size']
    device_type = eval_components['device_type']
    world_mesh = eval_components['world_mesh']

    pp_mesh = None
    pp_size = None
    if parallel_dims.pp_enabled and stages:
        pp_mesh = world_mesh["pp"]
        pp_size = pp_mesh.size()

    # Set model to evaluation mode
    for model in model_parts:
        model.eval()

    # Get train context for consistent behavior between training and evaluation
    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    eval_losses = []
    eval_perplexities = []

    if hasattr(eval_data_loader, 'reset'):
        eval_data_loader.reset()

    eval_iterator = iter(eval_data_loader)
    is_eval_exhausted = torch.zeros(world_size, dtype=torch.bool, device=device_type)

    with PipelineEphemeralContext(stages) if parallel_dims.pp_enabled and stages else contextlib.nullcontext():
        with torch.no_grad():
            for _ in range(job_config.evaluation.num_samples):
                try:
                    eval_batch = next(eval_iterator)
                    if eval_batch == "end":
                        is_eval_exhausted[rank] = True
                    torch.distributed.all_reduce(is_eval_exhausted)
                    if torch.any(is_eval_exhausted):
                        logger.info(f"Rank {rank}: Evaluation data exhausted. Ending evaluation.")
                        break
                except StopIteration:
                    logger.warning("Evaluation data exhausted before reaching num_samples. Ending evaluation.")
                    break

                # Extract batch data
                try:
                    input_ids, labels = eval_batch['input_ids'], eval_batch['labels']
                except (TypeError, KeyError):
                    input_ids, labels = eval_batch[0], eval_batch[1]

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)

                attention_mask = None
                if isinstance(eval_batch, dict) and 'attention_mask' in eval_batch:
                    attention_mask = eval_batch['attention_mask'].to(device_type)

                document_ids = None
                if isinstance(eval_batch, dict) and 'document_ids' in eval_batch:
                    document_ids = eval_batch['document_ids'].to(device_type)

                # Set state tensors (important for both PP and non-PP paths)
                state.set_state_tensors(
                    document_ids=document_ids,
                    attention_mask=attention_mask,
                    batch_size=job_config.evaluation.batch_size,
                    n_microbatches=job_config.experimental.pipeline_parallel_microbatches
                )

                # Set up context parallel if enabled
                if parallel_dims.cp_enabled:
                    cp_mesh = world_mesh["cp"]

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
                    cp_buffers.extend([m.freqs_cis for m in model_parts])
                    cp_seq_dims.extend([0 for _ in model_parts])

                    optional_context_parallel_ctx = utils.create_context_parallel_ctx(
                        cp_mesh=cp_mesh,
                        cp_buffers=cp_buffers,
                        cp_seq_dims=cp_seq_dims,
                        cp_no_restore_buffers=cp_no_restore_buffers,
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                else:
                    optional_context_parallel_ctx = None

                # Get reference model logits if enabled
                reference_logits = None
                if job_config.reference_model.enabled and reference_model is not None:
                    with torch.no_grad():
                        reference_logits = reference_model(input_ids, attention_mask)
                        # Set reference logits in state for context parallel
                        if parallel_dims.cp_enabled:
                            state.set_state_tensors(reference_logits=reference_logits)

                # Handle pipeline parallel and standard forward paths
                with train_context(optional_context_parallel_ctx):
                    if parallel_dims.pp_enabled and stages:
                        # Use the unified pipeline_forward implementation
                        output, has_last_stage = pipeline_forward(
                            stages=stages,
                            pp_size=pp_size,
                            inputs=input_ids,
                            mask=attention_mask,
                            stages_initialized=stages_initialized,
                        )

                        # Initialize loss with zeros
                        loss = torch.zeros(1, dtype=torch.float32, device=device_type)

                        # Compute loss only if we have the last stage
                        if has_last_stage and output is not None:
                            loss = loss_fn(output, labels, reference_logits, document_ids)
                    else:
                        # Standard non-pipeline forward pass
                        logits = model_parts[0](input_ids, mask=attention_mask)

                        # Apply the loss function
                        loss = loss_fn(logits, labels, reference_logits, document_ids)

                # Calculate perplexity from loss
                perplexity = torch.exp(loss)

                # Aggregate metrics across data parallel ranks
                if parallel_dims.dp_enabled or parallel_dims.cp_enabled:
                    # Use the combined dp_cp mesh for reduction
                    loss_val = utils.dist_mean(loss.detach(), world_mesh["dp_cp"])
                    perplexity_val = utils.dist_mean(perplexity.detach(), world_mesh["dp_cp"])
                else:
                    loss_val = loss.item()
                    perplexity_val = perplexity.item()

                eval_losses.append(loss_val)
                eval_perplexities.append(perplexity_val)

    # Calculate average metrics
    avg_loss, avg_perplexity = 0, 0
    if len(eval_losses) > 0:
        avg_loss = sum(eval_losses) / len(eval_losses)
        avg_perplexity = sum(eval_perplexities) / len(eval_perplexities)

    logger.info(f"Evaluation at step {current_step}: Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.4f}")

    # Log metrics
    metrics = {
        "eval/loss": avg_loss,
        "eval/perplexity": avg_perplexity,
        "eval/num_samples": len(eval_losses),
    }
    metric_logger.log(metrics, step=current_step)

    # Set model back to training mode
    for model in model_parts:
        model.train()

    return avg_loss, avg_perplexity

if __name__ == "__main__":
    # pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
