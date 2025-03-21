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
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._op_schema import OpStrategy, OpSchema, PlacementList, RuntimeSchemaInfo
from torch.distributed.tensor.experimental._attention import (
    _templated_ring_attention,
    _scaled_dot_product_ring_efficient_attention,
    _scaled_dot_product_ring_flash_attention,
    _CausalBehavior,
    _cp_options,
    _create_rotater,
    _SDPAMerger,
    _is_causal_behavior,
    _maybe_wait, _RotateMethod, context_parallel_unshard, _partial_update,
)
from torchtitan import state

logger = logging.getLogger(__name__)

aten = torch.ops.aten


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

    # Determine which mask section to return based on iteration and rank
    if i == 0:
        # First iteration: Self-attention for this rank's tokens
        left = local_mask[:, :, :, first_k_start:first_k_end]
        right = local_mask[:, :, :, second_k_start:second_k_end]

        # Concatenate horizontally to form the full mask
        result_mask = torch.cat([left, right], dim=3)
    elif i <= rank:
        # For i <= rank: Process all queries with first chunk of keys from rank (rank-i)
        result_mask = local_mask[:, :, :, first_k_start:first_k_end]
    else:  # i > rank
        # For i > rank: Only use the second half of queries against keys from source_rank
        # Extract mask for second chunk of queries x both chunks of keys
        bottom_left = local_mask[:, :, q_split:, first_k_start:first_k_end]
        bottom_right = local_mask[:, :, q_split:, second_k_start:second_k_end]

        # Concatenate horizontally to match the key dimension structure
        result_mask = torch.cat([bottom_left, bottom_right], dim=3)

    return result_mask

def monkey_patch_context_parallel_attention():
    """
    Monkey-patches PyTorch's context parallelism implementation to support attention masks
    from the global state object instead of direct parameters.

    Returns:
        callable: A function to restore the original implementations.
    """
    # Store original methods for reference and restoration
    import torch.distributed.tensor.experimental._attention as attn_module
    import torch.distributed.tensor._ops._matrix_ops as _matrix_ops
    original_sdp_efficient = attn_module._scaled_dot_product_ring_efficient_attention
    original_sdp_flash = attn_module._scaled_dot_product_ring_flash_attention
    original_templated_ring = attn_module._templated_ring_attention
    original_templated_ring_backward = attn_module._templated_ring_attention_backward
    original_sdpa_handler = attn_module._sdpa_handler
    original_sdpa_strategy = _matrix_ops.scaled_dot_product_efficient_attention_strategy
    original_sdpa_backward_strategy = _matrix_ops.scaled_dot_product_efficient_attention_backward_strategy

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
            is_causal=True,
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
    # Note: There is a slight numerical discrepancy between cp and dp in the forward pass. The discrepancy was significantly
    # reduced by changing attn_bias to use -1e9 instead of "inf." However, it wasn't fully eliminated, and should be looked
    # into in the future. The discrepancy is, loosely, about 4x the discrepancy between 1-rank and 4-rank dp_shard logits.
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
        with torch.compile compatibility.
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
                q, k, v, partial = query.chunk(2, dim=2)[1], key, value, True

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
                # Looks like using -1e9 instead of "inf" is more numerically stable.
                if current_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(current_mask.logical_not(), float(-1e9))
                else:
                    # If mask is already a float-based mask, convert to query dtype
                    attn_bias = current_mask.to(q.dtype)

                # Add as attn_bias parameter which is what efficient attention expects
                kwargs_to_pass["attn_bias"] = attn_bias
            else:
                # No mask available, use None for attn_bias
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

            sdpa_merger.step(out, logsumexp, partial)

        return (*sdpa_merger.results(), *rest)

    def patched_templated_ring_attention_backward(mesh, seq_dim, op, grad_out, grad_out_name, query, key, value,
                                                  out, logsumexp, is_causal, **kwargs):
        """Patched backward pass for templated ring attention with robust gradient handling"""
        # Use stderr to ensure logs are visible in distributed environment
        # Log current state for debugging
        is_causal = True
        import sys
        pg = mesh.get_group() if hasattr(mesh, "get_group") else mesh
        rank = dist.get_rank(pg)
        size = dist.get_world_size(pg)

        # Always use float32 for accumulation
        grad_query = torch.zeros_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)

        # Convert inputs to float32 for stability
        # print(f"DEBUG: query type {query.dtype}")
        query32 = query.to(torch.float32)
        key32 = key.contiguous().to(torch.float32)
        value32 = value.contiguous().to(torch.float32)
        grad_out32 = grad_out.to(torch.float32)
        out32 = out.to(torch.float32)
        logsumexp32 = logsumexp.to(torch.float32)

        # Create rotaters
        kv_rotater = _create_rotater(pg, 2)
        dkv_rotater = _create_rotater(pg, 2, method=_RotateMethod.ALL_TO_ALL)

        # For NaN checking
        def has_nan(tensor):
            return torch.isnan(tensor).any() or torch.isinf(tensor).any()

        # Get attention mask
        attn_bias = kwargs.get("attn_bias", None)
        state_mask = state.ATTENTION_MASK if hasattr(state, 'ATTENTION_MASK') else None
        mask_to_use = attn_bias if attn_bias is not None else state_mask

        # Process all iterations
        for i in range(size):
            # Exchange key-value information
            if i > 0:
                buffer = kv_rotater.next_buffer()
                pointer = 0
                key32 = buffer[pointer:pointer + key32.numel()].reshape(key32.shape)
                pointer += key32.numel()
                value32 = buffer[pointer:pointer + value32.numel()].reshape(value32.shape)

            if i < (size - 1):
                next_kv = torch.cat([key32.flatten(), value32.flatten()])
                kv_rotater.exchange_buffers(next_kv)

            # Determine causal behavior
            is_causal_behavior = _is_causal_behavior(rank, size, i, is_causal)

            if is_causal_behavior != _CausalBehavior.SKIP:
                # Set up tensors for this iteration
                if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                    q, k, v, out_, dout, lse = (query32, key32, value32, out32, grad_out32, logsumexp32)
                elif i <= rank:
                    q, k, v, out_, dout, lse = (
                        query32,
                        key32.chunk(2, dim=seq_dim)[0],
                        value32.chunk(2, dim=seq_dim)[0],
                        out32,
                        grad_out32,
                        logsumexp32,
                    )
                else:
                    q, k, v, out_, dout, lse = (
                        query32.chunk(2, dim=seq_dim)[1],
                        key32,
                        value32,
                        out32.chunk(2, dim=seq_dim)[1],
                        grad_out32.chunk(2, dim=seq_dim)[1],
                        logsumexp32.chunk(2, dim=seq_dim)[1].contiguous(),
                    )

                # Get mask chunk
                chunk_mask = None
                if mask_to_use is not None:
                    try:
                        chunk_mask = _get_mask_chunk(mask_to_use, q, i, rank, size)
                        # print(f"DEBUG: Using chunked mask for iter {i} with shape {chunk_mask.shape}", file=sys.stderr)
                    except Exception as e:
                        print(f"ERROR in mask chunking: {e}", file=sys.stderr)

                # Set up kwargs for op
                local_kwargs = kwargs.copy()
                local_kwargs[grad_out_name] = dout

                if chunk_mask is not None:
                    attn_bias = torch.zeros_like(chunk_mask, dtype=torch.float32)
                    if chunk_mask.dtype == torch.bool:
                        attn_bias.masked_fill_(chunk_mask.logical_not(), float("-inf"))
                    else:
                        attn_bias = chunk_mask.to(torch.float32)
                    local_kwargs["attn_bias"] = attn_bias


                # Call backward op with sanitized inputs
                # Replace NaN in inputs

                # Call backward with float32 precision
                grad_query_, grad_key_, grad_value_, *rest = op(
                    query=q, key=k, value=v, out=out_, logsumexp=lse,
                    is_causal=is_causal_behavior.value, **local_kwargs
                )

            else:
                # Skip iteration
                grad_query_ = torch.zeros_like(query32, dtype=torch.float32)
                grad_key_ = torch.zeros_like(key32, dtype=torch.float32)
                grad_value_ = torch.zeros_like(value32, dtype=torch.float32)

            # Update gradients with stable operations
            ROUND_ROBIN_CYCLE = 2
            if i == 0:
                # Direct addition with sanitization
                grad_key = grad_key + grad_key_
                grad_value =grad_value + grad_value_
            else:
                # Exchange gradients
                next_grad_kv = dkv_rotater.next_buffer()

                # Extract and prepare gradients
                pointer = 0
                grad_key = next_grad_kv[pointer:pointer + grad_key.numel()].reshape(grad_key.shape)
                pointer += grad_key.numel()
                grad_value = next_grad_kv[pointer:pointer + grad_value.numel()].reshape(grad_value.shape)

                # Replace any NaN/Inf
                grad_key = grad_key
                grad_value = grad_value


                if i <= rank and _cp_options.enable_load_balance:
                    grad_key = _partial_update(
                        grad_key,
                        grad_key_,
                        dim=seq_dim,
                        n_chunks=ROUND_ROBIN_CYCLE,
                        idx=0,
                        add=True,
                    )
                    grad_value = _partial_update(
                        grad_value,
                        grad_value_,
                        dim=seq_dim,
                        n_chunks=ROUND_ROBIN_CYCLE,
                        idx=0,
                        add=True,
                    )
                else:
                    grad_key = grad_key + grad_key_
                    grad_value = grad_value + grad_value_

            # Exchange gradients with sanitization
            next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
            dkv_rotater.exchange_buffers(next_grad_kv)

            # Update query gradient with special handling
            if i <= rank or not _cp_options.enable_load_balance:
                grad_query += grad_query_
            else:
                grad_query = _partial_update(
                    grad_query,
                    grad_query_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=1,
                    add=True,
                )

        # Final processing with careful conversion
        next_grad_kv = dkv_rotater.next_buffer()
        next_grad_kv = next_grad_kv

        # Extract final gradients
        grad_key = next_grad_kv[:grad_key.numel()].reshape(grad_key.shape)
        grad_value = next_grad_kv[grad_value.numel():].reshape(grad_value.shape)

        # Final sanitization and conversion to original dtype
        grad_query = grad_query.to(query.dtype)
        grad_key = grad_key.to(key.dtype)
        grad_value = grad_value.to(value.dtype)

        return (grad_query, grad_key, grad_value, *rest)

    @_matrix_ops.register_op_strategy(
        aten._scaled_dot_product_efficient_attention.default,
        schema_info=RuntimeSchemaInfo(4),
    )
    def patched_scaled_dot_product_efficient_attention_strategy(
            mesh: DeviceMesh, op_schema: OpSchema
    ) -> OpStrategy:
        # Original implementation with modifications for attn_bias
        q_input_strategy = op_schema.args_schema[0]
        assert isinstance(q_input_strategy, OpStrategy)

        has_attn_bias = op_schema.args_schema[3] is not None
        compute_log_sumexp = op_schema.args_schema[4]

        single_mesh_dim_strategies: list[PlacementList] = []

        # Full replication strategy
        all_replicate: PlacementList = [
            Replicate(),
            Replicate(),
            None,
            None,
            Replicate(),
            Replicate(),
            Replicate(),
        ]
        if has_attn_bias:
            all_replicate.append(Replicate())  # attn bias

        # Context Parallelism: shards on the sequence dim
        sequence_dim_sharding = [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # philox_seed
            None,  # philox_offset
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]

        # Add attention bias sharding if present
        if has_attn_bias:
            sequence_dim_sharding.append(Shard(2))  # attn_bias also sharded on seq dim

        # Add strategies to the list
        single_mesh_dim_strategies.append(sequence_dim_sharding)
        single_mesh_dim_strategies.append(all_replicate)

        # Tensor parallelism: shards on the heads dim
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        logsumexp_sharding = Shard(1) if compute_log_sumexp else Replicate()

        num_heads_dim_sharding = [
            output_sharding,
            logsumexp_sharding,
            None,
            None,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
        ]
        if has_attn_bias:
            num_heads_dim_sharding.append(Shard(1))

        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        return _matrix_ops.expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,
        )

    @_matrix_ops.register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
    def patched_scaled_dot_product_efficient_attention_backward_strategy(
            mesh: DeviceMesh, op_schema: OpSchema
    ) -> OpStrategy:
        q_input_strategy = op_schema.args_schema[1]
        assert isinstance(q_input_strategy, OpStrategy)

        # Check if attention bias is present
        has_attn_bias = op_schema.args_schema[4] is not None

        single_mesh_dim_strategies = []

        # Fully replicated strategy
        all_replicate: PlacementList = [Replicate()] * (12 + has_attn_bias)
        if not has_attn_bias:
            all_replicate[3] = None  # grad bias is None if attn_bias is not present
        single_mesh_dim_strategies.append(all_replicate)

        # Tensor parallelism (heads dimension) strategy
        grad_output_sharding = Shard(1)
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        logsumexp_sharding = Shard(1)
        grad_qkv_sharding = Shard(1)
        grad_bias_sharding = Shard(1) if has_attn_bias else None

        num_heads_dim_sharding: PlacementList = [
            grad_qkv_sharding,  # grad_q
            grad_qkv_sharding,  # grad_k
            grad_qkv_sharding,  # grad_v
            grad_bias_sharding,  # grad_bias (or None)
            grad_output_sharding,  # grad_output
            qkv_sharding,  # q
            qkv_sharding,  # k
            qkv_sharding,  # v
            # Position for optional attn_bias
            output_sharding,  # output
            logsumexp_sharding,  # logsumexp
        ]

        # Input sharding of attn_bias on heads dim if present
        if has_attn_bias:
            num_heads_dim_sharding.insert(8, Shard(1))  # attn_bias sharded on heads dim

        # Add remaining scalar tensor inputs
        num_heads_dim_sharding.extend([Replicate(), Replicate()])  # philox_seed, philox_offset

        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        # Context Parallelism (sequence dimension) strategy
        seq_dim_sharding: PlacementList = [
            Shard(2),  # grad_q
            Shard(2),  # grad_k
            Shard(2),  # grad_v
            None,  # grad_bias
            Shard(2),  # grad_output
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
            Shard(2),  # output
            Shard(2),  # logsumexp
        ]

        # Insert attn_bias sharding on sequence dim if present
        if has_attn_bias:
            seq_dim_sharding.insert(8, Shard(2))

        # Add remaining tensors
        seq_dim_sharding.extend([Replicate(), Replicate()])   # philox_seed, philox_offset

        single_mesh_dim_strategies.append(seq_dim_sharding)

        return _matrix_ops.expand_to_full_mesh_op_strategy(
            mesh,
            op_schema,
            single_mesh_dim_strategies,
            input_index=4,
        )

    # Apply all patches
    attn_module._scaled_dot_product_ring_efficient_attention = patched_sdp_efficient_attention
    attn_module._scaled_dot_product_ring_flash_attention = patched_sdp_flash_attention
    attn_module._templated_ring_attention = patched_templated_ring_attention
    attn_module._templated_ring_attention_backward = patched_templated_ring_attention_backward
    _matrix_ops.scaled_dot_product_efficient_attention_strategy = patched_scaled_dot_product_efficient_attention_strategy
    _matrix_ops.scaled_dot_product_efficient_attention_backward_strategy = patched_scaled_dot_product_efficient_attention_backward_strategy

    # logger.info("Successfully monkey-patched context parallelism attention functions with state-based masking")

    # Return a function to restore the original implementations if needed
    def restore_original():
        attn_module._scaled_dot_product_ring_efficient_attention = original_sdp_efficient
        attn_module._scaled_dot_product_ring_flash_attention = original_sdp_flash
        attn_module._templated_ring_attention = original_templated_ring
        attn_module._templated_ring_attention_backward = original_templated_ring_backward
        attn_module._sdpa_handler = original_sdpa_handler
        _matrix_ops.scaled_dot_product_efficient_attention_strategy = original_sdpa_strategy
        _matrix_ops.scaled_dot_product_efficient_attention_backward_strategy = original_sdpa_backward_strategy

        @_matrix_ops.register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
        def _restore_strategy(mesh, op_schema):
            return original_sdpa_strategy(mesh, op_schema)

        @_matrix_ops.register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
        def _restore_backward_strategy(mesh, op_schema):
            return original_sdpa_backward_strategy(mesh, op_schema)
        # logger.info("Restored original context parallelism attention functions")

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