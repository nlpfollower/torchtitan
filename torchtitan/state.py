from typing import Dict, Optional, Tuple
import torch

DOCUMENT_IDS = None
REFERENCE_LOGITS = None
ATTENTION_MASK = None
SEQUENCE_HASH_MAP = {}  # Maps sequence hash → microbatch index
MICROBATCH_SIZE = None
N_MICROBATCHES = None


def set_state_tensors(document_ids=None, reference_logits=None, attention_mask=None, labels=None, batch_size=None,
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
