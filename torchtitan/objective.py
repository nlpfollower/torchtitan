from enum import Enum
from typing import Tuple

import torch
import torch.nn.functional as F
from aiohttp.web_routedef import static
from torch import nn

from torchtitan.datasets import build_tokenizer


class Objective:
    @staticmethod
    def get_loss_function(loss_type):
        if loss_type == "default":
            return Objective.default_loss
        elif loss_type == "classification":
            return Objective.classification_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    @staticmethod
    def default_loss(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    @staticmethod
    def classification_loss(logits: torch.Tensor, labels: torch.Tensor, debug=False) -> torch.Tensor:
        # Shift logits and labels so we predict token n+1 from token n
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Use built-in ignore_index. CrossEntropy already excludes -100 labels from loss.
        loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,  # Skip the -100 tokens
            reduction="mean"  # or "sum" if you want control over scaling
        )

        return loss

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
        chosen_counts, rejected_counts = [], []

        for i in range(batch_size):
            unique_docs, counts = torch.unique(document_ids[i][document_ids[i] != -1], return_counts=True)
            batch_chosen_counts = counts[unique_docs % 2 == 0]
            batch_rejected_counts = counts[unique_docs % 2 == 1]

            chosen_counts.append(batch_chosen_counts)
            rejected_counts.append(batch_rejected_counts)

            max_chosen = max(max_chosen, batch_chosen_counts.max().item())
            max_rejected = max(max_rejected, batch_rejected_counts.max().item())

        # Initialize lists for chosen and rejected log probabilities
        policy_chosen_logps = []
        policy_rejected_logps = []
        reference_chosen_logps = []
        reference_rejected_logps = []

        # Unpack the samples
        for i in range(batch_size):
            chosen_doc_ids = document_ids[i][chosen_mask[i]]
            rejected_doc_ids = document_ids[i][rejected_mask[i]]

            unique_chosen_docs, chosen_counts_per_doc = torch.unique(chosen_doc_ids, return_counts=True)
            unique_rejected_docs, rejected_counts_per_doc = torch.unique(rejected_doc_ids, return_counts=True)

            chosen_splits = torch.split(policy_log_probs[i, chosen_mask[i]], chosen_counts_per_doc.tolist())
            rejected_splits = torch.split(policy_log_probs[i, rejected_mask[i]], rejected_counts_per_doc.tolist())

            ref_chosen_splits = torch.split(reference_log_probs[i, chosen_mask[i]], chosen_counts_per_doc.tolist())
            ref_rejected_splits = torch.split(reference_log_probs[i, rejected_mask[i]],
                                              rejected_counts_per_doc.tolist())

            for split, ref_split in zip(chosen_splits, ref_chosen_splits):
                if average_logprobs:
                    non_zero_mask = split != 0
                    policy_chosen_logps.append(
                        split[non_zero_mask].mean() if non_zero_mask.any() else torch.tensor(0.0, device=device))
                    reference_chosen_logps.append(
                        ref_split[non_zero_mask].mean() if non_zero_mask.any() else torch.tensor(0.0, device=device))
                else:
                    policy_chosen_logps.append(split.sum())
                    reference_chosen_logps.append(ref_split.sum())

            for split, ref_split in zip(rejected_splits, ref_rejected_splits):
                if average_logprobs:
                    non_zero_mask = split != 0
                    policy_rejected_logps.append(
                        split[non_zero_mask].mean() if non_zero_mask.any() else torch.tensor(0.0, device=device))
                    reference_rejected_logps.append(
                        ref_split[non_zero_mask].mean() if non_zero_mask.any() else torch.tensor(0.0, device=device))
                else:
                    policy_rejected_logps.append(split.sum())
                    reference_rejected_logps.append(ref_split.sum())

        return (
            torch.stack(policy_chosen_logps),
            torch.stack(policy_rejected_logps),
            torch.stack(reference_chosen_logps),
            torch.stack(reference_rejected_logps)
        )
