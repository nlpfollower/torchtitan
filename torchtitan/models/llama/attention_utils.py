import os
import pickle
from collections import defaultdict
from typing import Union, Callable, Optional

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask as create_block_causal_mask_flex,
    flex_attention, create_block_mask,
)

flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

def create_block_document_causal_mask(document_ids: torch.Tensor):
    batch_size, max_seq_len = document_ids.shape
    device = document_ids.device

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        non_padding = document_ids[b, q_idx] != -1
        return causal_mask & document_mask & non_padding

    return create_block_mask(
        mask_mod,
        batch_size,
        None,
        max_seq_len,
        max_seq_len,
        device=device,
    )

def create_document_causal_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """
    Creates a causal attention mask that respects document boundaries.

    Args:
        document_ids: (batch_size, seq_len) tensor of document IDs, with -1 for padding

    Returns:
        (batch_size, 1, seq_len, seq_len) boolean tensor where True allows attention
    """
    batch_size, seq_len = document_ids.shape
    device = document_ids.device

    # Create causal mask (lower triangular)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    # Create document matching mask
    # Expand dims for broadcasting: (batch, 1, seq_len) -> (batch, seq_len, seq_len)
    doc_ids_q = document_ids.unsqueeze(2)  # (batch, seq_len, 1)
    doc_ids_k = document_ids.unsqueeze(1)  # (batch, 1, seq_len)
    doc_match = (doc_ids_q == doc_ids_k)

    # Create padding mask
    non_padding = document_ids.unsqueeze(-1) != -1  # (batch, seq_len, 1)

    # Combine all masks
    mask = causal & doc_match & non_padding

    # Add singleton head dimension
    return mask.unsqueeze(1)

# We cannot do nested compile, but flex attention only has perf benefits
# when compiled. To insulate it from the compiler, we wrap it with
# compiler.disable so that it can be used regardless of whether the model
# is compiled or not, and flex attention always remains compiled.
@torch.compiler.disable(recursive=False)
def compile_friendly_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
) -> torch.Tensor:
    return flex_attention_compiled(q, k, v, block_mask=block_mask)

_MaskType = Union[torch.Tensor, BlockMask]

def sdpa_or_flex_attention() -> Callable:
    def _attention_call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[_MaskType] = None,
    ) -> torch.Tensor:
        layer_idx = getattr(torch, '_current_layer_idx', -1)

        if isinstance(mask, BlockMask):
            return compile_friendly_flex_attention(q, k, v, block_mask=mask)
        else:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                # Support for 2D mask in SDPA
                if mask is not None:
                    output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

                return output

    return _attention_call