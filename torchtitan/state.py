from typing import Dict, Optional, Tuple
import torch

DOCUMENT_IDS = None
REFERENCE_LOGITS = None
ATTENTION_MASK = None
SEQUENCE_HASH_MAP = {}  # Maps sequence hash → microbatch index
MICROBATCH_SIZE = None
N_MICROBATCHES = None


def compute_sequence_hash(input_ids: torch.Tensor) -> torch.Tensor:
    """Generate unique identifier for each sequence by summing all token IDs."""
    return input_ids.sum(dim=1).cpu() if isinstance(input_ids, torch.Tensor) else None


def set_state_tensors(document_ids=None, reference_logits=None, attention_mask=None, labels=None):
    """Initialize global state with direct hash-to-position mapping."""
    global DOCUMENT_IDS, REFERENCE_LOGITS, ATTENTION_MASK, SEQUENCE_HASH_MAP

    # Store full tensors
    if document_ids is not None:
        DOCUMENT_IDS = document_ids
    if reference_logits is not None:
        REFERENCE_LOGITS = reference_logits
    if attention_mask is not None:
        ATTENTION_MASK = attention_mask

    # Build direct hash-to-position mapping
    if labels is not None:
        SEQUENCE_HASH_MAP = {}
        # Compute all sequence hashes at once
        sequence_hashes = compute_sequence_hash(labels)

        # Assert that hash_val size matches input_ids size on the first dimension
        assert sequence_hashes.size(0) == labels.size(0), (
            f"Size mismatch: hash_val size {sequence_hashes.size(0)} does not match "
            f"input_ids size on first dim {labels.size(0)}"
        )

        # Map each sequence hash directly to its position in the original batch
        for j in range(labels.size(0)):
            hash_val = sequence_hashes[j].item()
            SEQUENCE_HASH_MAP[hash_val] = j  # Direct mapping to position


def get_sliced_tensor_by_indices(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Get tensor slice based on explicit indices."""
    if tensor is None or indices is None:
        return tensor

    # Convert indices tensor to list of integers for indexing
    if indices.dim() == 2 and indices.size(1) == 1:
        indices = indices.squeeze(1)

    # Handle case where indices is a single value
    if indices.dim() == 0:
        return tensor[indices.item()].unsqueeze(0)

    # Get the slice using the explicit indices
    return tensor[indices]


def get_sliced_tensor_by_hash(tensor: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    """Get tensor slice based on sequence hash mapping."""
    if tensor is None or sequence is None:
        return tensor

    # Compute hashes for current sequence
    sequence_hashes = compute_sequence_hash(sequence)

    # Find matching indices
    indices = []
    for i, hash_val in enumerate(sequence_hashes):
        idx = SEQUENCE_HASH_MAP.get(hash_val.item())
        if idx is not None:
            indices.append(idx)

    # Return sliced tensor if indices found, otherwise return original
    if not indices:
        raise ValueError("Failed to find matching indices for sequence in hash map")

        # Sort indices and check if they're continuous
    indices.sort()
    if not all(indices[i] == indices[0] + i for i in range(len(indices))):
        raise ValueError(f"Non-continuous indices detected: {indices}")

    return tensor[indices]