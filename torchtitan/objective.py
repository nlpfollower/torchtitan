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
    def classification_loss(logits: torch.Tensor, labels: torch.Tensor, debug = False) -> torch.Tensor:
        # Find the positions of non-negative labels
        shifted_labels = labels[:, 1:].contiguous()
        shifted_logits = logits[:, :-1, :].contiguous()

        # Find the positions of non-negative labels in the shifted tensor
        valid_positions = shifted_labels != -100

        # Apply mask to shifted logits and labels
        valid_logits = shifted_logits[valid_positions]
        valid_labels = shifted_labels[valid_positions]

        if debug:
            _, predicted_tokens = torch.max(valid_logits, dim=-1)
            # Build the tokenizer
            tok = build_tokenizer("llama", "models/Llama3.2-3B-Instruct/tokenizer.model")

            # Decode the predicted tokens
            decoded_tokens = tok.decode(predicted_tokens.tolist())
            decoded_labels = tok.decode(valid_labels.tolist())
            print(f"Predicted: {decoded_tokens}")
            print(f"Labels: {decoded_labels}")

        # Compute negative log likelihood loss
        loss = F.cross_entropy(valid_logits, valid_labels)

        return loss