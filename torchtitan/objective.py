from enum import Enum
from typing import Tuple

import torch
import torch.nn.functional as F
from aiohttp.web_routedef import static
from torch import nn

from torchtitan.logging import logger


class Objective:
    @staticmethod
    def get_loss_function(loss_type):
        if loss_type == "default":
            return Objective.default_loss
        elif loss_type == "classification":
            return Objective.classification_loss
        elif loss_type == "classification_with_packing":
            return Objective.classification_loss_with_packing
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    @staticmethod
    def default_loss(pred, labels, document_ids):
        return F.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    @staticmethod
    def classification_loss(logits: torch.Tensor, labels: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
        # Shift logits and labels so we predict token n+1 from token n
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Use built-in ignore_index. CrossEntropy already excludes -100 labels from loss.
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fn(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )

        return loss

    @staticmethod
    def classification_loss_with_packing(logits: torch.Tensor, labels: torch.Tensor,
                                         document_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, vocab_size = logits.shape

        # Shift tensors
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        shifted_document_ids = document_ids[:, 1:].contiguous()

        # Flatten tensors
        flat_logits = shifted_logits.reshape(-1, vocab_size)
        flat_labels = shifted_labels.reshape(-1)

        # Compute loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fn(flat_logits, flat_labels).view(batch_size, -1)

        # Compute mean loss for each document in each batch
        batch_losses = []
        for b in range(batch_size):
            unique_docs = torch.unique(shifted_document_ids[b])
            unique_docs = unique_docs[unique_docs != -1]  # Remove padding document ID
            doc_losses = []

            for doc_id in unique_docs:
                # Create mask for the current document
                doc_mask = (shifted_document_ids[b] == doc_id)
                label_mask = (shifted_labels[b] != -100)
                combined_mask = doc_mask & label_mask

                # Apply the combined mask to the loss
                masked_loss = loss[b][combined_mask]
                if masked_loss.numel() > 0:
                    doc_loss = masked_loss.mean()
                    doc_losses.append(doc_loss)

            if doc_losses:
                batch_loss = torch.stack(doc_losses).mean()
                batch_losses.append(batch_loss)

        try:
            return torch.stack(batch_losses).mean()
        except RuntimeError as e:
            if "stack expects each tensor to be equal size" in str(e):
                # Log the error and return a default tensor
                logger.info(f"Error in classification_loss_with_packing: {e}")
                logger.info(f"batch_losses: {batch_losses}")
                return torch.tensor([0.0], device=logits.device, dtype=logits.dtype)
            else:
                # Re-raise the exception if it's not the specific error we're handling
                raise

class ReferenceObjective:
    @staticmethod
    def get_loss_function(loss_type):
        if loss_type == "dpo":
            return ReferenceObjective.dpo_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    @staticmethod
    def dpo_loss(policy_logits, reference_logits, labels, document_ids, beta=0.1):
        # Compute log probabilities first
        policy_log_probs = ReferenceObjective._compute_log_probs(policy_logits, labels)
        reference_log_probs = ReferenceObjective._compute_log_probs(reference_logits, labels)
        document_ids = document_ids[:, 1:].contiguous()

        # Unpack the batch
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps = ReferenceObjective._unpack_batch(
            policy_log_probs, reference_log_probs, document_ids)

        # Compute DPO loss
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        losses = -F.logsigmoid(beta * (policy_logratios - reference_logratios))

        return losses.mean()

    @staticmethod
    def _compute_log_probs(logits, labels):
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        batch_size, seq_len, vocab_size = logits.shape
        gathered_log_probs = -loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        gathered_log_probs = gathered_log_probs.view(batch_size, seq_len)

        return gathered_log_probs

    @staticmethod
    def _unpack_batch(policy_log_probs, reference_log_probs, document_ids, average_logprobs=False):
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
