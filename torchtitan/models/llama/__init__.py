# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.datasets import build_hf_data_loader
from torchtitan.datasets.tokenizer import TikTokenizer
from torchtitan.loss import cross_entropy_loss
from torchtitan.models.llama.model import Transformer, TransformerModelArgs
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.train_spec import register_train_spec, TrainSpec

from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=8, n_heads=16, rope_theta=500000
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.5,
        multiple_of=256,
        rope_theta=500000,
    ),
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.0,
        multiple_of=256,
        rope_theta=500000,
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}