from enum import Enum
from typing import Tuple

import torch
import torch.nn.functional as F
from aiohttp.web_routedef import static

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
    def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                 reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor,
                 beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(beta * logits)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards