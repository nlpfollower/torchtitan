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
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                token_loss = loss_fn(
                    shifted_logits.reshape(-1, shifted_logits.size(-1)),
                    shifted_labels.reshape(-1)
                ).view(batch_size, -1)

                # Handle reference logits if provided
                if reference_logits is not None:
                    # Shift reference logits
                    shifted_ref_logits = reference_logits[:, :-1, :].contiguous()

                    # Compute token-level losses for reference logits
                    reference_token_loss = loss_fn(
                        shifted_ref_logits.reshape(-1, shifted_ref_logits.size(-1)),
                        shifted_labels.reshape(-1)
                    ).view(batch_size, -1)

            # Call the loss function with token losses instead of logits
            return loss_fn(token_loss, labels, reference_token_loss, document_ids)

        return wrapped_loss

    @staticmethod
    def _compute_cp_token_loss(logits, labels, cp_info):
        """
        Computes token-level cross entropy efficiently for context parallel shards.

        For causal language modeling with CP:
        - Each token at position i predicts the token at position i+1
        - The last CP rank must exclude its last logit (no label to predict)
        - We calculate exact indices to map between sharded logits and full labels

        Args:
            logits: Sharded logits [batch_size, shard_seq_len, vocab_size]
            labels: Full unsharded labels [batch_size, full_seq_len]
            cp_info: Dict containing 'rank' and 'world_size'

        Returns:
            token_loss: Token-level cross entropy loss [batch_size, shard_seq_len or shard_seq_len-1]
            token_mask: Boolean mask of valid tokens (not -100) [batch_size, shard_seq_len or shard_seq_len-1]
        """
        batch_size, shard_seq_len, vocab_size = logits.shape
        cp_rank, cp_world_size = cp_info['rank'], cp_info['world_size']

        # Calculate sequence partition for this rank
        full_seq_len = labels.shape[1]
        shard_size = (full_seq_len + cp_world_size - 1) // cp_world_size  # Ceiling division

        # Calculate logit positions for this rank
        logit_start_idx = cp_rank * shard_size
        logit_end_idx = min(logit_start_idx + shard_size, full_seq_len)

        # For the last CP rank, we need to exclude the last logit position
        # since there's no corresponding label to predict
        if cp_rank == cp_world_size - 1 and logit_end_idx == full_seq_len:
            logit_end_idx -= 1

        # Calculate the corresponding label positions (shift by +1 for causal modeling)
        label_start_idx = logit_start_idx + 1
        label_end_idx = logit_end_idx + 1

        # For the last rank, we need to truncate the logits tensor
        if cp_rank == cp_world_size - 1:
            # Only use logits that have corresponding labels
            usable_seq_len = logit_end_idx - logit_start_idx
            logits = logits[:, :usable_seq_len].contiguous()

        # Extract exactly the labels that correspond to this shard's logits
        target_labels = labels[:, label_start_idx:label_end_idx].contiguous()

        # Compute token-level cross entropy
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        token_loss = loss_fn(
            logits.reshape(-1, vocab_size),
            target_labels.reshape(-1)
        ).view(batch_size, -1)

        # Store indices for unsharding
        indices = {
            'logit_start': logit_start_idx,
            'logit_end': logit_end_idx,
            'label_start': label_start_idx,
            'label_end': label_end_idx
        }

        return token_loss, indices

    @staticmethod
    def _unshard_loss_parallel(token_loss, cp_info, indices):
        """
        Unshards token-level losses across CP ranks with gradient support.

        Args:
            token_loss: Sharded token losses [batch_size, shard_seq_len]
            cp_info: Dict with CP info including 'rank', 'world_size', and 'group'
            indices: Dict with start/end indices for this shard

        Returns:
            unsharded_token_loss: Full sequence token losses [batch_size, seq_len-1]
        """
        cp_rank, cp_world_size, cp_group = cp_info['rank'], cp_info['world_size'], cp_info['group']
        logit_start, logit_end = indices['logit_start'], indices['logit_end']

        # Create custom autograd function for token loss unsharding with gradient support
        class LossParallelUnshard(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, rank, world_size, group, start_idx, end_idx, full_seq_len):
                # Ensure input is contiguous
                tensor = tensor.contiguous()

                # Gather sharded tensors from all ranks
                gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

                # Perform all-gather operation
                torch.distributed.all_gather(gathered_tensors, tensor, group=group)

                # Create empty tensor for full sequence
                full_tensor_shape = list(tensor.shape)
                full_tensor_shape[1] = full_seq_len - 1  # -1 because we're predicting shifts

                full_tensor = torch.zeros(full_tensor_shape, device=tensor.device, dtype=tensor.dtype)

                # Fill in the full tensor with gathered shards
                for i, shard in enumerate(gathered_tensors):
                    # Calculate shard boundaries
                    # world_size - 1 to round up
                    shard_size = (full_seq_len + world_size - 1) // world_size
                    shard_start = i * shard_size
                    shard_end = min(shard_start + shard_size, full_seq_len)

                    # For the last rank, adjust if needed
                    if i == world_size - 1 and shard_end == full_seq_len:
                        shard_end -= 1

                    # If shard is empty, skip
                    if shard_end <= shard_start:
                        continue

                    # Calculate corresponding shifted positions (for causal modeling)
                    target_start = shard_start
                    target_len = shard_end - shard_start

                    # Place this shard's data in the right position
                    slice_len = min(target_len, shard.shape[1])
                    full_tensor[:, target_start:target_start + slice_len] = shard[:, :slice_len]

                # Save context for backward
                ctx.rank = rank
                ctx.world_size = world_size
                ctx.group = group
                ctx.start_idx = start_idx
                ctx.end_idx = end_idx
                ctx.shard_size = end_idx - start_idx

                return full_tensor

            @staticmethod
            def backward(ctx, grad_output):
                # Extract saved context
                start_idx = ctx.start_idx
                end_idx = ctx.end_idx

                # Extract gradient for this rank's shard
                grad_for_shard = grad_output[:, start_idx:end_idx].contiguous()

                # Return gradient for input tensor, None for other arguments
                return grad_for_shard, None, None, None, None, None, None

        # Get full sequence length from labels
        full_seq_len = cp_info['full_seq_len']

        # Apply unsharding to token loss with gradient support
        unsharded_token_loss = LossParallelUnshard.apply(
            token_loss, cp_rank, cp_world_size, cp_group,
            logit_start, logit_end, full_seq_len
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
