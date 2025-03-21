from enum import Enum
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from aiohttp.web_routedef import static
from torch import nn

from torchtitan import state
from torchtitan.logging import logger


class Objective:
    @staticmethod
    def get_loss_function(loss_type):
        if loss_type == "default":
            loss_fn = Objective.default_loss
        elif loss_type == "classification_with_packing":
            loss_fn = Objective.classification_loss_with_packing
        elif loss_type == "dpo_with_packing":
            loss_fn = Objective.dpo_loss_with_packing
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        return Objective.wrap_loss_fn(loss_fn)

    @staticmethod
    def wrap_loss_fn(loss_fn):
        """
        Wraps a loss function to handle context parallel optimization.

        This wrapper transforms the interface from logits-based to loss-based for all loss functions:
        1. In standard mode: Compute token-level losses from logits
        2. In CP mode: Compute & unshard token-level losses from sharded logits

        All loss functions then operate uniformly on token-level losses rather than logits.
        """

        def wrapped_loss(logits, labels, reference_logits=None, document_ids=None, mb_index=None):
            # Handle state tensor retrieval for microbatching
            if reference_logits is None and mb_index is not None and state.REFERENCE_LOGITS is not None:
                if state.MICROBATCH_SIZE is not None:
                    start_idx = mb_index * state.MICROBATCH_SIZE
                    end_idx = start_idx + min(state.MICROBATCH_SIZE, labels.size(0))
                    reference_logits = state.REFERENCE_LOGITS[start_idx:end_idx]
                else:
                    reference_logits = state.REFERENCE_LOGITS

            if document_ids is None and mb_index is not None and state.DOCUMENT_IDS is not None:
                if state.MICROBATCH_SIZE is not None:
                    start_idx = mb_index * state.MICROBATCH_SIZE
                    end_idx = start_idx + min(state.MICROBATCH_SIZE, labels.size(0))
                    document_ids = state.DOCUMENT_IDS[start_idx:end_idx]
                else:
                    document_ids = state.DOCUMENT_IDS

            # Prepare containers for token losses
            token_loss = None
            reference_token_loss = None

            # Check if context parallel is enabled
            is_cp_enabled = hasattr(state, 'CP_WORLD_SIZE') and state.CP_WORLD_SIZE > 1

            if is_cp_enabled:
                # Context parallel mode - compute & unshard token losses
                cp_info = {
                    'rank': state.CP_RANK,
                    'world_size': state.CP_WORLD_SIZE,
                    'group': state.CP_GROUP if hasattr(state, 'CP_GROUP') else None,
                    'full_seq_len': labels.shape[1]
                }

                # Compute token-level CE loss on sharded policy logits
                sharded_token_loss, indices = Objective._compute_cp_token_loss(logits, labels, cp_info)

                # Unshard token losses - much cheaper than unsharding logits
                token_loss = Objective._unshard_loss_parallel(
                    sharded_token_loss, cp_info, indices
                )

                # Handle reference logits if provided
                if reference_logits is not None:
                    # Compute token-level loss on reference logits
                    sharded_ref_loss, ref_indices = Objective._compute_cp_token_loss(reference_logits, labels, cp_info)

                    # Unshard reference token losses
                    reference_token_loss = Objective._unshard_loss_parallel(
                        sharded_ref_loss, cp_info, ref_indices
                    )
            else:
                # Standard mode - compute token losses directly
                batch_size = logits.shape[0]

                # Shift for causal modeling
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = labels[:, 1:].contiguous()

                # Compute token-level losses for policy logits
                ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                token_loss = ce_loss_fn(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),
                    shifted_labels.reshape(-1)
                ).view(batch_size, -1)

                # Handle reference logits if provided
                if reference_logits is not None:
                    # Shift reference logits
                    shifted_ref_logits = reference_logits[:, :-1, :].contiguous()

                    # Compute token-level losses for reference logits
                    reference_token_loss = ce_loss_fn(
                        shifted_ref_logits.reshape(-1, shifted_ref_logits.size(-1)),
                        shifted_labels.reshape(-1)
                    ).view(batch_size, -1)

            # Call the loss function with token losses instead of logits
            return loss_fn(token_loss, labels, reference_token_loss, document_ids)

        return wrapped_loss

    @staticmethod
    def _compute_cp_token_loss(logits, labels, cp_info):
        """
        Computes token-level cross entropy efficiently for context parallel shards with round-robin pattern.

        In the round-robin pattern:
        - Each rank r gets two non-contiguous chunks: (r, 2*world_size-1-r)
        - Each rank's logits tensor already contains these two concatenated chunks
        - We need to map these logits to the correct label positions for causal modeling

        Args:
            logits: Sharded logits [batch_size, shard_seq_len, vocab_size]
            labels: Full unsharded labels [batch_size, full_seq_len]
            cp_info: Dict containing 'rank', 'world_size', and 'full_seq_len'

        Returns:
            token_loss: Token-level cross entropy loss [batch_size, shard_seq_len]
            indices: Dict with mapping information for unsharding
        """
        batch_size, shard_seq_len, vocab_size = logits.shape
        cp_rank, cp_world_size = cp_info['rank'], cp_info['world_size']
        full_seq_len = cp_info['full_seq_len']

        # Calculate chunk size for the round-robin pattern
        # Each rank gets 2 chunks, each of size chunk_size
        chunk_size = full_seq_len // (2 * cp_world_size)

        # Calculate the positions of this rank's two chunks in the full sequence
        chunk1_start = cp_rank * chunk_size  # First chunk starts at rank*chunk_size
        chunk2_start = (2 * cp_world_size - 1 - cp_rank) * chunk_size  # Second chunk position

        # For causal modeling, label positions are shifted by +1
        chunk1_label_start = chunk1_start + 1
        chunk2_label_start = chunk2_start + 1

        # In round-robin pattern, rank 0 always gets the last chunk as its second chunk
        # Split logits into the two chunks they represent
        first_chunk_size = chunk_size
        second_chunk_size = chunk_size

        # Adjust second chunk size for rank 0 which contains the last position
        if cp_rank == 0:
            second_chunk_size -= 1  # Last position has no next token for prediction

        # Split the logits into the two chunks
        logits_chunk1 = logits[:, :first_chunk_size].contiguous()
        logits_chunk2 = logits[:, first_chunk_size:first_chunk_size + second_chunk_size].contiguous()

        # Extract the corresponding label sections for each chunk
        labels_chunk1 = labels[:, chunk1_label_start:chunk1_label_start + first_chunk_size].contiguous()
        labels_chunk2 = labels[:, chunk2_label_start:chunk2_label_start + second_chunk_size].contiguous()

        # Compute token-level cross entropy for each chunk
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        # Process first chunk
        token_loss_chunk1 = loss_fn(
            logits_chunk1.reshape(-1, vocab_size),
            labels_chunk1.reshape(-1)
        ).view(batch_size, -1)

        # Process second chunk
        token_loss_chunk2 = loss_fn(
            logits_chunk2.reshape(-1, vocab_size),
            labels_chunk2.reshape(-1)
        ).view(batch_size, -1)

        # Concatenate the token losses from both chunks
        token_loss = torch.cat([token_loss_chunk1, token_loss_chunk2], dim=1)

        # Store indices for unsharding
        indices = {
            'chunk1_start': chunk1_start,
            'chunk1_end': chunk1_start + first_chunk_size,
            'chunk2_start': chunk2_start,
            'chunk2_end': chunk2_start + second_chunk_size,
            'chunk_size': chunk_size
        }

        return token_loss, indices

    def _unshard_loss_parallel(token_loss, cp_info, indices):
        """
        Unshards token-level losses across CP ranks with gradient support for round-robin pattern.

        Args:
            token_loss: Sharded token losses [batch_size, shard_seq_len]
            cp_info: Dict with CP info including 'rank', 'world_size', and 'group'
            indices: Dict with chunk mapping information

        Returns:
            unsharded_token_loss: Full sequence token losses [batch_size, seq_len-1]
        """
        cp_rank, cp_world_size, cp_group = cp_info['rank'], cp_info['world_size'], cp_info['group']
        full_seq_len = cp_info['full_seq_len']
        chunk_size = indices['chunk_size']

        # Create custom autograd function for token loss unsharding with gradient support
        class RoundRobinLossUnshard(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, rank, world_size, group, indices, full_seq_len):
                # Extract indices
                chunk1_start = indices['chunk1_start']
                chunk1_end = indices['chunk1_end']
                chunk2_start = indices['chunk2_start']
                chunk2_end = indices['chunk2_end']
                chunk_size = indices['chunk_size']

                # Get the actual shape of the input tensor
                batch_size, actual_shard_len = tensor.shape

                # Split the token loss into its two constituent chunks
                first_chunk_size = chunk1_end - chunk1_start
                second_chunk_size = chunk2_end - chunk2_start

                loss_chunk1 = tensor[:, :first_chunk_size].contiguous()
                loss_chunk2 = tensor[:, first_chunk_size:].contiguous()

                # Pad both chunks to ensure consistent sizes across ranks
                padded_chunk1 = torch.zeros(
                    (batch_size, chunk_size),
                    device=tensor.device,
                    dtype=tensor.dtype
                )
                padded_chunk2 = torch.zeros(
                    (batch_size, chunk_size),
                    device=tensor.device,
                    dtype=tensor.dtype
                )

                padded_chunk1[:, :first_chunk_size] = loss_chunk1
                padded_chunk2[:, :second_chunk_size] = loss_chunk2

                # Concatenate the padded chunks for all_gather
                gather_tensor = torch.cat([padded_chunk1, padded_chunk2], dim=1).contiguous()

                # Gather sharded tensors from all ranks
                gathered_size = gather_tensor.size()
                gathered_list = [torch.zeros_like(gather_tensor) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_list, gather_tensor, group=group)

                # Create empty tensor for full sequence result (minus 1 for causal prediction)
                full_token_loss = torch.zeros(
                    (batch_size, full_seq_len - 1),
                    device=tensor.device,
                    dtype=tensor.dtype
                )

                # Extract and place all chunks correctly
                for i, gathered_tensor in enumerate(gathered_list):
                    # Split the gathered tensor back into its two chunks
                    g_chunk1 = gathered_tensor[:, :chunk_size]
                    g_chunk2 = gathered_tensor[:, chunk_size:]

                    # Calculate positions for this rank's chunks
                    rank_chunk1_start = i * chunk_size
                    rank_chunk1_end = rank_chunk1_start + chunk_size

                    rank_chunk2_start = (2 * world_size - 1 - i) * chunk_size
                    rank_chunk2_end = rank_chunk2_start + chunk_size

                    # Special case: last position in sequence has no next token to predict
                    # In round-robin, only rank 0's second chunk can contain the last position
                    if i == 0 and rank_chunk2_end == full_seq_len:
                        rank_chunk2_end = full_seq_len - 1

                    # Copy valid parts of each chunk to the full tensor
                    if rank_chunk1_end > rank_chunk1_start:
                        valid_size = rank_chunk1_end - rank_chunk1_start
                        full_token_loss[:, rank_chunk1_start:rank_chunk1_end] = g_chunk1[:, :valid_size]

                    if rank_chunk2_end > rank_chunk2_start:
                        valid_size = rank_chunk2_end - rank_chunk2_start
                        full_token_loss[:, rank_chunk2_start:rank_chunk2_end] = g_chunk2[:, :valid_size]

                # Save context for backward
                ctx.rank = rank
                ctx.chunk1_start = chunk1_start
                ctx.chunk1_end = chunk1_end
                ctx.chunk2_start = chunk2_start
                ctx.chunk2_end = chunk2_end
                ctx.tensor_shape = tensor.shape

                return full_token_loss

            @staticmethod
            def backward(ctx, grad_output):
                # Extract saved context
                rank = ctx.rank
                chunk1_start = ctx.chunk1_start
                chunk1_end = ctx.chunk1_end
                chunk2_start = ctx.chunk2_start
                chunk2_end = ctx.chunk2_end
                original_shape = ctx.tensor_shape

                # Extract gradient for this rank's chunks
                grad_chunk1 = grad_output[:, chunk1_start:chunk1_end].contiguous()
                grad_chunk2 = grad_output[:, chunk2_start:chunk2_end].contiguous()

                # Concatenate the two chunk gradients
                grad_for_shard = torch.cat([grad_chunk1, grad_chunk2], dim=1)

                # Ensure gradient matches the original tensor shape
                if grad_for_shard.shape[1] != original_shape[1]:
                    # Pad or truncate to match original shape
                    if grad_for_shard.shape[1] < original_shape[1]:
                        # Pad
                        padded_grad = torch.zeros(
                            original_shape,
                            device=grad_for_shard.device,
                            dtype=grad_for_shard.dtype
                        )
                        padded_grad[:, :grad_for_shard.shape[1]] = grad_for_shard
                        grad_for_shard = padded_grad
                    else:
                        # Truncate
                        grad_for_shard = grad_for_shard[:, :original_shape[1]]

                # Return gradient for input tensor, None for other arguments
                return grad_for_shard, None, None, None, None, None

        # Apply unsharding with gradient support
        unsharded_token_loss = RoundRobinLossUnshard.apply(
            token_loss, cp_rank, cp_world_size, cp_group, indices, full_seq_len
        )

        return unsharded_token_loss

    @staticmethod
    def _reduce_token_loss(token_loss, labels):
        """
        Reduces token-level loss properly accounting for valid token positions.

        Args:
            token_loss: Token-level losses [batch_size, seq_len-1]
            labels: Full labels tensor [batch_size, seq_len]

        Returns:
            Properly reduced loss scalar
        """
        # Create mask for valid token positions (not -100)
        # Since we're in causal modeling, we need shifted labels (labels[:, 1:])
        shifted_labels = labels[:, 1:].contiguous()
        token_mask = (shifted_labels != -100)

        # Count valid tokens
        valid_tokens = token_mask.sum()

        if valid_tokens > 0:
            # Average loss over valid tokens only
            return (token_loss * token_mask.float()).sum() / valid_tokens
        else:
            # Handle edge case with no valid tokens
            return torch.tensor(0.0, device=token_loss.device, dtype=token_loss.dtype)

    @staticmethod
    def default_loss(token_loss, labels, reference_token_loss=None, document_ids=None):
        """
        Default cross-entropy loss implementation.

        Now accepts token-level losses directly instead of logits.
        """
        return Objective._reduce_token_loss(token_loss, labels)

    @staticmethod
    def classification_loss_with_packing(token_loss: torch.Tensor, labels: torch.Tensor,
                                         reference_token_loss: Optional[torch.Tensor] = None,
                                         document_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classification loss with document-aware packing and CP support.

        Computes per-document losses and averages them, respecting document boundaries.
        """
        # Prepare inputs differently based on CP vs. standard mode
        # Get shifted labels and document IDs for the full sequence
        shifted_labels = labels[:, 1:].contiguous()
        shifted_document_ids = document_ids[:, 1:].contiguous()

        # Create mask for valid token positions
        token_mask = (shifted_labels != -100)

        # Calculate per-document losses
        batch_size = token_loss.shape[0]
        batch_losses = []

        # Process by batch and document
        for b in range(batch_size):
            unique_docs = torch.unique(shifted_document_ids[b])
            unique_docs = unique_docs[unique_docs != -1]  # Remove padding document ID
            doc_losses = []

            for doc_id in unique_docs:
                # Document + valid token mask
                doc_mask = (shifted_document_ids[b] == doc_id) & token_mask[b]

                # Get losses for this document
                doc_token_losses = token_loss[b][doc_mask]

                if doc_token_losses.numel() > 0:
                    doc_loss = doc_token_losses.mean()
                    doc_losses.append(doc_loss)

            if doc_losses:
                batch_loss = torch.stack(doc_losses).mean()
                batch_losses.append(batch_loss)

        # Compute final loss
        try:
            return torch.stack(batch_losses).mean()
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                logger.info(f"Error in classification_loss_with_packing: {e}")
                return torch.tensor([0.0], device=token_loss.device, dtype=token_loss.dtype)
            else:
                raise

    @staticmethod
    def dpo_loss_with_packing(token_loss: torch.Tensor, labels: torch.Tensor,
                              reference_token_loss: Optional[torch.Tensor] = None,
                              document_ids: Optional[torch.Tensor] = None,
                              beta: float = 0.1) -> torch.Tensor:
        """
        Direct Preference Optimization (DPO) loss.

        Now accepts token-level losses directly instead of logits.

        Args:
            token_loss: Policy model token-level losses [batch_size, seq_len-1]
            labels: Target labels [batch_size, seq_len]
            reference_token_loss: Reference model token-level losses [batch_size, seq_len-1]
            document_ids: Document IDs for identifying chosen vs rejected samples
            beta: KL divergence weight (default: 0.1)
        """
        if reference_token_loss is None:
            raise ValueError("DPO loss requires reference_token_loss")

        if document_ids is None:
            raise ValueError("DPO loss requires document_ids")

        # Convert token losses to log probs
        policy_log_probs = -token_loss
        reference_log_probs = -reference_token_loss

        # Get shifted document IDs
        shifted_document_ids = document_ids[:, 1:].contiguous()

        # Unpack for DPO calculation
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps = (
            Objective._unpack_batch_dpo(policy_log_probs, reference_log_probs, shifted_document_ids)
        )

        # Compute DPO loss
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * (policy_logratios - reference_logratios))
        return losses.mean()

    @staticmethod
    def _unpack_batch_dpo(policy_log_probs, reference_log_probs, document_ids, average_logprobs=False):
        batch_size, seq_len = document_ids.shape
        device = policy_log_probs.device

        # Create masks for chosen and rejected samples
        chosen_mask = (document_ids % 2 == 0) & (document_ids != -1)
        rejected_mask = (document_ids % 2 == 1) & (document_ids != -1)

        # Process each batch item separately
        max_chosen, max_rejected = 0, 0
        policy_chosen_logps, policy_rejected_logps = [], []
        reference_chosen_logps, reference_rejected_logps = [], []

        for i in range(batch_size):
            batch_chosen_mask = chosen_mask[i]
            batch_rejected_mask = rejected_mask[i]

            if batch_chosen_mask.any():
                chosen_doc_ids = document_ids[i][batch_chosen_mask]
                unique_chosen_docs, chosen_counts = torch.unique(chosen_doc_ids, return_counts=True)
                max_chosen = max(max_chosen, chosen_counts.max().item())

                chosen_splits = torch.split(policy_log_probs[i, batch_chosen_mask], chosen_counts.tolist())
                ref_chosen_splits = torch.split(reference_log_probs[i, batch_chosen_mask], chosen_counts.tolist())

                for split, ref_split in zip(chosen_splits, ref_chosen_splits):
                    if average_logprobs:
                        policy_chosen_logps.append(split.mean())
                        reference_chosen_logps.append(ref_split.mean())
                    else:
                        policy_chosen_logps.append(split.sum())
                        reference_chosen_logps.append(ref_split.sum())
            else:
                # Handle empty chosen samples
                policy_chosen_logps.append(torch.tensor(0.0, device=device))
                reference_chosen_logps.append(torch.tensor(0.0, device=device))

            if batch_rejected_mask.any():
                rejected_doc_ids = document_ids[i][batch_rejected_mask]
                unique_rejected_docs, rejected_counts = torch.unique(rejected_doc_ids, return_counts=True)
                max_rejected = max(max_rejected, rejected_counts.max().item())

                rejected_splits = torch.split(policy_log_probs[i, batch_rejected_mask], rejected_counts.tolist())
                ref_rejected_splits = torch.split(reference_log_probs[i, batch_rejected_mask], rejected_counts.tolist())

                for split, ref_split in zip(rejected_splits, ref_rejected_splits):
                    if average_logprobs:
                        policy_rejected_logps.append(split.mean())
                        reference_rejected_logps.append(ref_split.mean())
                    else:
                        policy_rejected_logps.append(split.sum())
                        reference_rejected_logps.append(ref_split.sum())
            else:
                # Handle empty rejected samples
                policy_rejected_logps.append(torch.tensor(0.0, device=device))
                reference_rejected_logps.append(torch.tensor(0.0, device=device))

        return (
            torch.stack(policy_chosen_logps),
            torch.stack(policy_rejected_logps),
            torch.stack(reference_chosen_logps),
            torch.stack(reference_rejected_logps)
        )
