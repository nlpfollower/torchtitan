from typing import Dict, Optional, Tuple, Any
import torch

# Tensor state variables
DOCUMENT_IDS = None
REFERENCE_LOGITS = None
ATTENTION_MASK = None
SEQUENCE_HASH_MAP = {}  # Maps sequence hash → microbatch index

# Microbatch information
MICROBATCH_SIZE = None
N_MICROBATCHES = None

# Context Parallel information
CP_RANK = None
CP_WORLD_SIZE = None
CP_GROUP = None
CP_MESH = None


def set_state_tensors(document_ids=None, reference_logits=None, attention_mask=None, batch_size=None,
                      n_microbatches=None):
    """Initialize global state with tensor data and microbatch information."""
    global DOCUMENT_IDS, REFERENCE_LOGITS, ATTENTION_MASK, MICROBATCH_SIZE, N_MICROBATCHES

    # Store full tensors
    if document_ids is not None:
        DOCUMENT_IDS = document_ids
    if reference_logits is not None:
        REFERENCE_LOGITS = reference_logits
    if attention_mask is not None:
        ATTENTION_MASK = attention_mask

    # Store microbatch information
    if batch_size is not None and n_microbatches is not None:
        MICROBATCH_SIZE = batch_size // n_microbatches
        N_MICROBATCHES = n_microbatches


def set_cp_info(cp_rank=None, cp_world_size=None, cp_group=None, cp_mesh=None):
    """Initialize global state with context parallel information."""
    global CP_RANK, CP_WORLD_SIZE, CP_GROUP, CP_MESH

    if cp_rank is not None:
        CP_RANK = cp_rank
    if cp_world_size is not None:
        CP_WORLD_SIZE = cp_world_size
    if cp_group is not None:
        CP_GROUP = cp_group
    if cp_mesh is not None:
        CP_MESH = cp_mesh
