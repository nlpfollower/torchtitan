# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module for context parallelism related utilities and monkey patches.
"""

import contextlib
import logging
import os
import pickle
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import (
    _templated_ring_attention,
    _scaled_dot_product_ring_efficient_attention,
    _scaled_dot_product_ring_flash_attention,
    _CausalBehavior,
    _cp_options,
    _create_rotater,
    _SDPAMerger,
    _is_causal_behavior,
    _maybe_wait, _RotateMethod, context_parallel_unshard,
)
from torchtitan import state

logger = logging.getLogger(__name__)

aten = torch.ops.aten

ATTENTION_TRACING = False
ATTENTION_OUTPUTS = defaultdict(list)
TRACE_DIR = None

def enable_attention_tracing(output_dir):
    """Enable tracing of attention operations"""
    global ATTENTION_TRACING, ATTENTION_OUTPUTS, TRACE_DIR
    ATTENTION_TRACING = True
    ATTENTION_OUTPUTS.clear()

    # Add container for detailed tensor data
    ATTENTION_OUTPUTS["cp_detailed"] = []

    TRACE_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def disable_attention_tracing():
    """Disable tracing and save collected outputs"""
    global ATTENTION_TRACING
    ATTENTION_TRACING = False
    save_attention_outputs()


def save_attention_outputs():
    """Save collected attention outputs to file"""
    if not TRACE_DIR:
        return

    output_path = os.path.join(TRACE_DIR, f"attention_outputs_rank{torch.distributed.get_rank()}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(ATTENTION_OUTPUTS, f)

    print(f"Saved {len(ATTENTION_OUTPUTS)} attention traces to {output_path}")

def _get_mask_chunk(local_mask, q_chunk, i, rank, size):
    """
    Extract the appropriate chunk of attention mask for the current SDPA operation with round-robin scheduling.

    Args:
        local_mask: The 4D global attention mask (batch, heads, seq/size, seq)
        q_chunk: Current query chunk
        k_chunk: Current key chunk
        i: Current iteration in the ring rotation
        rank: Rank of the current process
        size: World size for context parallel
        is_causal: Whether causal masking is used
        seq_dim: Sequence dimension (default: 2)

    Returns:
        The appropriate chunk of the mask
    """
    if local_mask is None:
        return None

    # Assert requirements for this implementation
    assert _cp_options.enable_load_balance, "This implementation requires load balancing to be enabled"
    assert _cp_options.rotate_method == _RotateMethod.ALL_GATHER, "Only ALL_GATHER rotation method is supported"
    assert local_mask.dim() == 4, "Mask must be 4D (batch, heads, q_seq, kv_seq)"
    # Get the number of heads from the query tensor
    num_heads_q = q_chunk.size(1)

    # If mask has only 1 head but query has multiple, broadcast the mask
    if local_mask.size(1) == 1 and num_heads_q > 1:
        # Expand the mask to match the number of heads
        local_mask = local_mask.expand(-1, num_heads_q, -1, -1)

    # Calculate sequence length per rank
    full_seq_len = local_mask.size(3)
    seq_len_per_rank = full_seq_len // (2 * size)

    # Split point for query dimension - always at the midpoint
    q_split = local_mask.size(2) // 2

    # Source rank for keys in this iteration
    source_rank = (rank - i) % size

    # Key position calculations (reused across cases)
    # First chunk key positions for source rank
    first_k_start = source_rank * seq_len_per_rank
    first_k_end = (source_rank + 1) * seq_len_per_rank

    # Second chunk key positions for source rank (reversed in round-robin)
    second_k_start = (2 * size - 1 - source_rank) * seq_len_per_rank
    second_k_end = (2 * size - source_rank) * seq_len_per_rank

    if i == 0:
        # First iteration: Self-attention for this rank's tokens
        # Extract mask for all queries x all keys from this rank
        left = local_mask[:, :, :, first_k_start:first_k_end]
        right = local_mask[:, :, :, second_k_start:second_k_end]

        # Concatenate horizontally to form the full mask
        return torch.cat([left, right], dim=3)

    elif i <= rank:
        # For i <= rank: Process all queries with first chunk of keys from rank (rank-i)
        # Looking at the templated_ring_attention, we see it only uses key.chunk(2)[0]
        # So we only need the mask for the first chunk of keys
        return local_mask[:, :, :, first_k_start:first_k_end]

    else:  # i > rank
        # For i > rank: Only use the second half of queries against keys from source_rank
        # Extract mask for second chunk of queries x both chunks of keys
        bottom_left = local_mask[:, :, q_split:, first_k_start:first_k_end]
        bottom_right = local_mask[:, :, q_split:, second_k_start:second_k_end]

        # Concatenate horizontally to match the key dimension structure
        return torch.cat([bottom_left, bottom_right], dim=3)

def monkey_patch_context_parallel_attention():
    """
    Monkey-patches PyTorch's context parallelism implementation to support attention masks
    from the global state object instead of direct parameters.

    Returns:
        callable: A function to restore the original implementations.
    """
    # Store original methods for reference and restoration
    import torch.distributed.tensor.experimental._attention as attn_module
    original_sdp_efficient = attn_module._scaled_dot_product_ring_efficient_attention
    original_sdp_flash = attn_module._scaled_dot_product_ring_flash_attention
    original_templated_ring = attn_module._templated_ring_attention

    # Patched version of _scaled_dot_product_ring_efficient_attention that supports state-based masks
    def patched_sdp_efficient_attention(
            mesh: DeviceMesh,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_bias: Optional[torch.Tensor] = None,
            compute_log_sumexp: bool = True,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            *,
            scale: Optional[float] = None,
    ) -> tuple[torch.Tensor, ...]:
        # Allow both attn_bias parameter and state-based mask
        # Prefer explicitly passed attn_bias if available
        if not compute_log_sumexp:
            # CP requires compute_log_sumexp to be True because it always merges LSE
            compute_log_sumexp = True

        seq_dim = 2
        return patched_templated_ring_attention(
            mesh,
            seq_dim,
            aten._scaled_dot_product_efficient_attention,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            attn_bias=attn_bias,  # Pass through explicitly provided mask if available
            dropout_p=dropout_p,
            scale=scale,
            compute_log_sumexp=compute_log_sumexp,
        )

    # Patched version of _scaled_dot_product_ring_flash_attention that supports state-based masks
    def patched_sdp_flash_attention(
            mesh: DeviceMesh,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            return_debug_mask: bool = False,
            *,
            scale: Optional[float] = None,
            attn_mask: Optional[torch.Tensor] = None,  # Added support for attention mask
    ) -> tuple[torch.Tensor, ...]:
        if return_debug_mask:
            raise NotImplementedError("return_debug_mask is not supported yet")

        seq_dim = 2
        return patched_templated_ring_attention(
            mesh,
            seq_dim,
            aten._scaled_dot_product_flash_attention,
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            attn_mask=attn_mask,  # Pass through explicitly provided mask if available
        )

    # Patched version of _templated_ring_attention that handles state-based masks
    def patched_templated_ring_attention(
            mesh: DeviceMesh,
            seq_dim: int,
            op: Callable,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            is_causal: bool = False,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """
        Modified version of _templated_ring_attention that handles chunking of attention masks
        from either explicit parameters or the global state.
        """
        assert op == aten._scaled_dot_product_efficient_attention, "Only efficient attention is supported"

        if is_causal and (query.size(2) != key.size(2)):
            raise NotImplementedError(
                "is_causal requires the same query and context sequence lengths"
            )
        if not is_causal and _cp_options.enable_load_balance:
            raise RuntimeError("Load balancing requires `is_causal=True`.")

        if isinstance(mesh, dist.ProcessGroup):
            pg: Union[dist.ProcessGroup, list[dist.ProcessGroup]] = mesh
        else:
            pg = mesh.get_group()
        assert isinstance(pg, dist.ProcessGroup), "process group must be single dimension"
        rank = dist.get_rank(pg)
        size = dist.get_world_size(pg)

        next_kv = None

        # Without making key and value contiguous(), the lose curve is bad.
        key = key.contiguous()
        value = value.contiguous()

        sdpa_merger = _SDPAMerger(_cp_options.convert_to_f32, seq_dim=seq_dim)

        rest: list[Any]
        out: torch.Tensor
        logsumexp: torch.Tensor

        rotater = _create_rotater(pg, 2)

        # Extract explicit mask parameters if provided
        attn_bias = kwargs.pop("attn_bias", None)
        attn_mask = kwargs.pop("attn_mask", None)

        # Determine which mask to use (priority: explicit > state)
        explicit_mask = attn_bias if attn_bias is not None else attn_mask
        global_mask = state.ATTENTION_MASK if hasattr(state, 'ATTENTION_MASK') else None

        # Use explicit mask if provided, otherwise try global mask from state
        mask_to_use = explicit_mask if explicit_mask is not None else global_mask

        for i in range(size):
            if i > 0:
                # Wait for the kv from the (cp_rank - 1) rank.
                next_kv = rotater.next_buffer()
                key = next_kv[: key.numel()].reshape(key.shape)
                value = next_kv[key.numel():].reshape(value.shape)

            if i < (size - 1):
                # Send the k, v to the next rank
                next_kv = torch.cat([key.flatten(), value.flatten()])
                next_kv = rotater.exchange_buffers(next_kv)

            is_causal_behavior = _is_causal_behavior(
                rank=rank, world_size=size, i=i, is_causal=is_causal
            )

            # For a detailed understanding of the load balancing algorithm, see
            # Note [Context parallelism load balance algorithm for causal masking]
            if is_causal_behavior == _CausalBehavior.SKIP:
                # If i > rank and load balancing is not turned on.
                continue

            # Determine chunks of q, k, v based on the iteration and rank
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                # When local balance is enabled, we still need to do SDPA with
                # the both local chunks of q, k, v for the first iteration.
                q, k, v, partial = (query, key, value, False)
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SPDA, with only the first local chunk of the k, v.
                # Note that q, k, v, each contains two local chunks.
                ROUND_ROBIN_CYCLE = 2
                q, k, v, partial = (
                    query,
                    key.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                    value.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                    False,
                )
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SPDA with only the second half of the q, and update
                # only the the second part of logsumexp. So partial is True.
                # Note that q, k, v, each contains two chunks.
                ROUND_ROBIN_CYCLE = 2
                q_chunks = query.chunk(ROUND_ROBIN_CYCLE, dim=2)
                q, k, v, partial = q_chunks[1], key, value, True

            # Get the appropriate mask chunk for this operation
            current_mask = _get_mask_chunk(mask_to_use, q, i, rank, size)

            # Set up kwargs for the operation
            kwargs_to_pass = kwargs.copy()
            if current_mask is not None:
                # Get expected dimensions from query and key
                L, S = q.size(-2), k.size(-2)

                # Validate mask shape matches query/key dimensions
                mask_L, mask_S = current_mask.size(-2), current_mask.size(-1)
                assert mask_L == L, f"Mask query dimension {mask_L} doesn't match query dimension {L}"
                assert mask_S == S, f"Mask key dimension {mask_S} doesn't match key dimension {S}"

                # Create bias tensor with proper shape and dtype
                attn_bias = torch.zeros_like(current_mask, dtype=q.dtype)

                # Convert boolean mask to bias by filling masked positions with -inf
                if current_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(current_mask.logical_not(), float("-inf"))
                else:
                    # If mask is already a float-based mask, convert to query dtype
                    attn_bias = current_mask.to(q.dtype)

                # Add as attn_bias parameter which is what efficient attention expects
                kwargs_to_pass["attn_bias"] = attn_bias
            else:
                # Create a None tensor for attn_bias
                kwargs_to_pass["attn_bias"] = None

            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            out, logsumexp, *rest = op(
                q,
                k,
                v,
                is_causal=is_causal_behavior.value,
                **kwargs_to_pass,
            )

            # Trace the outputs
            if ATTENTION_TRACING:
                layer_idx = getattr(torch, '_current_layer_idx', -1)

                # Only trace detailed data for layer 0
                if layer_idx == 0:
                    ATTENTION_OUTPUTS["cp_detailed"].append({
                        "layer_idx": layer_idx,
                        "iteration": i,
                        "rank": rank,
                        "source_rank": (rank - i) % size,
                        "is_causal": is_causal_behavior.value,
                        "partial": partial,
                        "q": q.detach().cpu(),
                        "k": k.detach().cpu(),
                        "mask": current_mask.detach().cpu() if current_mask is not None else None
                    })

                # Keep regular output tracing for all layers
                ATTENTION_OUTPUTS["cp"].append({
                    "layer_idx": layer_idx,
                    "iteration": i,
                    "rank": rank,
                    "is_causal": is_causal_behavior.value,
                    "partial": partial,
                    "output_shape": out.shape,
                    "output": out.detach().cpu(),
                    "logsumexp": logsumexp.detach().cpu()
                })

            sdpa_merger.step(out, logsumexp, partial)

        results = sdpa_merger.results()

        # Record merged results if tracing is enabled
        if ATTENTION_TRACING:
            cp_output = results[0]
            cp_logsumexp = results[1]

            # Unshard the outputs to full sequence length
            # Store both sharded and unsharded versions
            unsharded_outputs = None
            if mesh is not None:
                try:
                    # Unshard along sequence dimension (2)
                    unsharded_outputs = context_parallel_unshard(mesh, [cp_output], [2])
                    unsharded_output = unsharded_outputs[0]

                    # Also unshard logsumexp if needed
                    unsharded_logsumexp = context_parallel_unshard(mesh, [cp_logsumexp], [1])[0]
                except Exception as e:
                    # Log error but continue with sharded outputs
                    print(f"Error unsharding output: {e}")

            ATTENTION_OUTPUTS["cp_merged"].append({
                "layer_idx": layer_idx,
                "rank": rank,
                "output_shape": cp_output.shape,
                "output": cp_output.detach().cpu(),
                "logsumexp": cp_logsumexp.detach().cpu(),
                "unsharded_output_shape": unsharded_output.shape if unsharded_outputs else None,
                "unsharded_output": unsharded_output.detach().cpu() if unsharded_outputs else None,
                "unsharded_logsumexp": unsharded_logsumexp.detach().cpu() if unsharded_outputs else None
            })

        return (*results, *rest)

    # Apply the patches
    attn_module._scaled_dot_product_ring_efficient_attention = patched_sdp_efficient_attention
    attn_module._scaled_dot_product_ring_flash_attention = patched_sdp_flash_attention
    attn_module._templated_ring_attention = patched_templated_ring_attention

    logger.info("Successfully monkey-patched context parallelism attention functions with state-based masking")

    # Return a function to restore the original implementations if needed
    def restore_original():
        attn_module._scaled_dot_product_ring_efficient_attention = original_sdp_efficient
        attn_module._scaled_dot_product_ring_flash_attention = original_sdp_flash
        attn_module._templated_ring_attention = original_templated_ring
        logger.info("Restored original context parallelism attention functions")

    return restore_original


@contextlib.contextmanager
def context_parallel_with_attention_mask():
    """
    Context manager to enable attention mask support in context parallelism.

    This temporarily applies the required monkey patches to support attention masks
    in context parallelism (both explicit and state-based), and restores the original
    implementations when exiting.

    Usage:
        with context_parallel_with_attention_mask():
            # Code that uses context parallelism with attention masks
    """
    restore_fn = monkey_patch_context_parallel_attention()
    try:
        yield
    finally:
        restore_fn()


def shard_attention_mask(attention_mask, cp_mesh):
    """
    Shard the attention mask according to the round-robin pattern required by CP.
    CRITICAL: This must exactly match the _RoundRobinLoadBalancer.shard() implementation.

    Args:
        attention_mask: The full attention mask tensor
        cp_mesh: The context parallel mesh

    Returns:
        Properly sharded attention mask for the current rank
    """
    if attention_mask is None:
        return None

    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()

    # Mask must be 4D (batch, heads, q_seq, kv_seq)
    assert attention_mask.dim() == 4, "Mask must be 4D (batch, heads, q_seq, kv_seq)"

    # For query dimension (dim=2), use the same round-robin pattern as for tensors
    q_seq_len = attention_mask.size(2)
    assert q_seq_len % (cp_size * 2) == 0, f"Sequence length {q_seq_len} must be divisible by 2*CP size {2 * cp_size}"

    # Shard the query dimension EXACTLY as done in _RoundRobinLoadBalancer.shard()
    chunks = attention_mask.chunk(cp_size * 2, dim=2)
    q_sharded_mask = torch.cat(
        (chunks[cp_rank], chunks[cp_size * 2 - cp_rank - 1]),
        dim=2
    )

    # For key dimension (dim=3), we need a full copy since attention is across all keys
    # but we could also apply the same pattern for dim=3 to be consistent
    k_seq_len = attention_mask.size(3)

    # The key is to use this exact mask with no further processing in _get_mask_chunk
    return q_sharded_mask