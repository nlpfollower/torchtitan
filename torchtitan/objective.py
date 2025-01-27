from enum import Enum

import torch
import torch.nn.functional as F

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

        if debug:
            # For debugging, pick out only valid positions
            valid_positions = shifted_labels != -100
            valid_logits = shifted_logits[valid_positions]
            valid_labels = shifted_labels[valid_positions]

            # Get the predicted token IDs
            _, predicted_tokens = torch.max(valid_logits, dim=-1)

            # Decode the predicted tokens
            tok = build_tokenizer("llama", "models/Llama3.2-3B-Instruct/tokenizer.model")
            decoded_tokens = tok.decode(predicted_tokens.tolist())
            decoded_labels = tok.decode(valid_labels.tolist())
            print(f"Predicted: {decoded_tokens}")
            print(f"Labels: {decoded_labels}")

        return loss