# Large prime modulus for finite field arithmetic

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.logging import logger
from torchtitan.utils import gather_dtensor

PRIME = (1 << 61) - 1

# Probably redundant, since the checksum is at most equal to the number of parameters.
def finite_field_add(a, b):
    """Addition in GF(2^61 - 1)."""
    return (a + b) % PRIME


def tensor_checksum(tensor):
    if tensor.numel() == 0:
        return torch.tensor(0, dtype=torch.float64, device=tensor.device)

    tensor = tensor.to(dtype=torch.float32, device=tensor.device)
    mean = tensor.mean()
    std = tensor.std()
    normalized = (tensor - mean) / (std + 1e-8)
    mapped = torch.sigmoid(normalized)
    checksum = mapped.sum()
    checksum += torch.sigmoid(mean) + torch.sigmoid(std)
    return checksum.to(dtype=torch.float64)

def combine_checksums(checksums):
    """Combine multiple checksums in a way that preserves uniqueness."""
    return sum(checksums) % PRIME


def checksum_model(model):
    rank = dist.get_rank()
    if rank != 0:
        return None, None

    checksums = {}
    total_checksum = 0

    model_parts = model.model_parts if hasattr(model, 'model_parts') else [model]

    for idx, part in enumerate(model_parts):
        for name, param in part.named_parameters():
                gathered_param = gather_dtensor(param.data)
                checksum = tensor_checksum(gathered_param)
                checksums[f"part_{idx}.{name}"] = checksum.item()
                total_checksum = finite_field_add(total_checksum, checksum)

    return total_checksum, checksums