import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pydevd_pycharm
import torch
import torch.distributed as dist
from datasets import load_dataset
from llama_models.llama3.api import RawMessage, ChatFormat, ToolPromptFormat
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.datasets.hh_dataset import HHDataset, build_hh_data_loader, packed_document_causal_mask
from torchtitan.datasets.tokenizer import build_tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger, init_logger
from torchtitan.models import llama3_configs, Transformer
from torchtitan.models.reference_model import build_reference_model
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import get_device_info, set_determinism


def setup_config():
    logger.info("Setting up test configuration")
    config = JobConfig()
    config.parse_args(['--model.tokenizer_path', 'models/Llama3.1-8B-Instruct/tokenizer.model'])
    return config


def setup_environment():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="nccl")
    device_type, _ = get_device_info()
    device = torch.device(f"{device_type}:0")
    set_determinism(None, device, seed=42)
    return device, device_type

def create_job_config():
    job_config = JobConfig()
    job_config.parse_args([])
    job_config.model.name = "llama3"
    job_config.model.flavor = "8B"
    job_config.model.tokenizer_path = "models/Llama3.1-8B-Instruct/tokenizer.model"
    job_config.job.dump_folder = "outputs"
    job_config.checkpoint.folder = "checkpoint"
    job_config.checkpoint.enable_checkpoint = True
    job_config.training.seq_len = 8192
    job_config.reference_model.checkpoint_path = "outputs/checkpoint/step-0"
    job_config.reference_model.data_parallel_shard_degree = 1
    job_config.reference_model.tensor_parallel_degree = 1
    job_config.reference_model.pipeline_parallel_degree = 1
    return job_config

def packed_preference_causal_mask(option_ids: torch.Tensor, document_ids: torch.Tensor):
    batch_size, max_seq_len = option_ids.shape
    option_ids = option_ids.to("cuda")
    document_ids = document_ids.to("cuda")

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        option_mask = torch.logical_not(torch.logical_and(option_ids[b, q_idx] == 2, option_ids[b, kv_idx] == 1))
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        non_padding = torch.logical_and(document_ids[b, q_idx] != -1, document_ids[b, kv_idx] != -1)
        return causal_mask & option_mask & document_mask & non_padding

    return create_block_mask(
        mask_mod,
        batch_size,
        None,
        max_seq_len,
        max_seq_len,
        device="cuda",
    )

def compare_logits_detailed(masked, unmasked, name, tokenizer, token_str = False):
    # Ensure both tensors are on the same device and have the same dtype
    unmasked = unmasked.to(masked.device, masked.dtype)

    # Compute differences
    diff = torch.abs(masked - unmasked)

    # Compute statistics
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    per_position_max = diff.max(dim=-1).values
    per_position_mean = diff.mean(dim=-1)

    logger.info(f"\n{name} Comparison:")
    logger.info(f"Overall - Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

    logger.info("Per-position statistics and next token predictions:")
    for pos in range(diff.size(0)):
        masked_next_token = masked[pos].argmax().item()
        unmasked_next_token = unmasked[pos].argmax().item()
        masked_next_token_str, unmasked_next_token_str = '',''
        if token_str:
            masked_next_token_str = tokenizer.decode([masked_next_token])
            unmasked_next_token_str = tokenizer.decode([unmasked_next_token])

        logger.info(f"Position {pos:3d} - "
                    f"Max diff: {per_position_max[pos].item():.6f}, "
                    f"Mean diff: {per_position_mean[pos].item():.6f}, "
                    f"Masked next: {masked_next_token_str} (id={masked_next_token}), "
                    f"Unmasked next: {unmasked_next_token_str} (id={unmasked_next_token})")

        if masked_next_token != unmasked_next_token:
            logger.warning(f"Token prediction mismatch at position {pos} in {name}!")


def test_sdpa_mask_equivalence():
    device, device_type = setup_environment()
    job_config = create_job_config()
    job_config.model.attention = "sdpa"
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    reference_model = build_reference_model(job_config, world_mesh, parallel_dims, tokenizer)
    chat_format = ChatFormat(tokenizer)

    def encode_prompt(messages):
        tokens = [tokenizer.bos_id]
        for message in messages:
            toks, _ = chat_format.encode_message(message, tool_prompt_format=ToolPromptFormat.json)
            tokens.extend(toks)
        return tokens

    def encode_chosen_rejected(message):
        return chat_format.encode_dialog_prompt([message]).tokens[1:]

    # Define messages
    messages = [
        RawMessage(role="system", content="Reply with a number only"),
        RawMessage(role="user", content="Sum up all the numbers you see."),
        RawMessage(role="user", content="5")
    ]

    # Encode sequences
    prompt = encode_prompt(messages)
    chosen = encode_chosen_rejected(RawMessage(role="user", content="12"))
    rejected = encode_chosen_rejected(RawMessage(role="user", content="10"))

    # Create the packed sequence
    all_tokens = prompt + chosen + [tokenizer.eos_id] + prompt + rejected
    tokens = torch.tensor(all_tokens).unsqueeze(0).to(device)

    # Create masks
    seq_len = len(all_tokens)
    split_point = len(prompt) + len(chosen) + 1

    # SDPA mask
    sdpa_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    sdpa_mask[:split_point, :split_point] = True
    sdpa_mask[split_point:, split_point:] = True
    sdpa_mask = sdpa_mask.tril()

    # Print the mask for verification
    logger.info("2D Attention Mask:")
    mask_np = sdpa_mask.cpu().numpy()
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    mask_str = np.array2string(mask_np, separator='', formatter={'bool': lambda x: '1' if x else '0'})
    logger.info(f"\n{mask_str}")

    chosen_mask = torch.tril(torch.ones((len(prompt) + len(chosen), len(prompt) + len(chosen)), dtype=torch.bool, device=device))
    rejected_mask = torch.tril(torch.ones((len(prompt) + len(rejected), len(prompt) + len(rejected)), dtype=torch.bool, device=device))

    # Masked forward pass
    with torch.no_grad():
        masked_logits = reference_model(tokens, sdpa_mask)

    # Unmasked forward passes for prompt+chosen and prompt+rejected
    with torch.no_grad():
        unmasked_logits_chosen = reference_model(torch.tensor(prompt + chosen).unsqueeze(0).to(device), chosen_mask)
        unmasked_logits_rejected = reference_model(torch.tensor(prompt + rejected).unsqueeze(0).to(device), rejected_mask)

    # Prepare slices for comparison
    masked_prompt_chosen = masked_logits[0, :len(prompt) + len(chosen)]
    masked_prompt_rejected = masked_logits[0, split_point:split_point + len(prompt) + len(rejected)]

    # Logit comparisons
    compare_logits_detailed(masked_prompt_chosen, unmasked_logits_chosen[0], "Prompt + Chosen", tokenizer)
    compare_logits_detailed(masked_prompt_rejected, unmasked_logits_rejected[0], "Prompt + Rejected", tokenizer)

    # Additional verification
    logger.info(f"Masked logits shape: {masked_logits.shape}")
    logger.info(f"Unmasked chosen logits shape: {unmasked_logits_chosen.shape}")
    logger.info(f"Unmasked rejected logits shape: {unmasked_logits_rejected.shape}")
    logger.info(f"Masked prompt+chosen shape: {masked_prompt_chosen.shape}")
    logger.info(f"Masked prompt+rejected shape: {masked_prompt_rejected.shape}")

    logger.info("SDPA mask equivalence test completed. Check the detailed comparisons above.")

def create_document_causal_mask(document_ids: torch.Tensor) -> torch.Tensor:
    seq_len = document_ids.size(0)
    mask = (document_ids.unsqueeze(1) == document_ids.unsqueeze(0)).tril()
    return mask


def pack_samples(samples: List[Tuple[List[int], List[int], List[int]]]) -> Tuple[
    torch.Tensor, torch.Tensor]:
    all_tokens = []
    document_ids = []
    current_doc_id = 0

    for prompt, chosen, rejected in samples:
        all_tokens.extend(prompt + chosen)
        document_ids.extend([current_doc_id] * (len(prompt) + len(chosen)))
        current_doc_id += 1

        all_tokens.extend(prompt + rejected)
        document_ids.extend([current_doc_id] * (len(prompt) + len(rejected)))
        current_doc_id += 1

    return torch.tensor(all_tokens, dtype=torch.long), torch.tensor(document_ids, dtype=torch.long)


def compare_masked_and_individual_passes(reference_model, masked_logits, input_ids, document_ids, labels, device):
    for b in range(input_ids.shape[0]):
        sample_input_ids = input_ids[b]
        sample_document_ids = document_ids[b]
        sample_labels = labels[b]

        unique_doc_ids = torch.unique(sample_document_ids)
        unique_doc_ids = unique_doc_ids[unique_doc_ids != -1]  # Remove padding document ID

        for doc_id in range(0, len(unique_doc_ids), 2):
            chosen_mask = (sample_document_ids == unique_doc_ids[doc_id])
            rejected_mask = (sample_document_ids == unique_doc_ids[doc_id + 1]) if doc_id + 1 < len(unique_doc_ids) else torch.zeros_like(chosen_mask)

            chosen_input = sample_input_ids[chosen_mask]
            rejected_input = sample_input_ids[rejected_mask]

            with torch.no_grad():
                chosen_logits = reference_model(chosen_input.unsqueeze(0), None)[0]
                rejected_logits = reference_model(rejected_input.unsqueeze(0), None)[0] if rejected_mask.any() else None

            masked_chosen = masked_logits[b, chosen_mask]
            masked_rejected = masked_logits[b, rejected_mask] if rejected_mask.any() else None

            diff_chosen = torch.abs(masked_chosen - chosen_logits)
            max_diff_chosen = diff_chosen.max().item()
            mean_diff_chosen = diff_chosen.mean().item()

            if rejected_logits is not None:
                diff_rejected = torch.abs(masked_rejected - rejected_logits)
                max_diff_rejected = diff_rejected.max().item()
                mean_diff_rejected = diff_rejected.mean().item()
                mismatch_rejected = (masked_rejected.argmax(dim=-1) != rejected_logits.argmax(dim=-1)).sum().item()
            else:
                max_diff_rejected = 0
                mean_diff_rejected = 0
                mismatch_rejected = 0

            mismatch_chosen = (masked_chosen.argmax(dim=-1) != chosen_logits.argmax(dim=-1)).sum().item()

            total_tokens = len(chosen_input) + (len(rejected_input) if rejected_logits is not None else 0)

            yield {
                "batch": b,
                "sample": doc_id // 2,
                "chosen_max_diff": max_diff_chosen,
                "chosen_mean_diff": mean_diff_chosen,
                "chosen_mismatches": mismatch_chosen,
                "rejected_max_diff": max_diff_rejected,
                "rejected_mean_diff": mean_diff_rejected,
                "rejected_mismatches": mismatch_rejected,
                "total_tokens": total_tokens
            }


def test_advanced_document_causal_mask(num_samples: int, max_seq_length: int, batch_size: int):
    device, device_type = setup_environment()
    job_config = create_job_config()
    job_config.model.attention = "sdpa"
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    reference_model = build_reference_model(job_config, world_mesh, parallel_dims, tokenizer)

    dataloader = build_hh_data_loader(tokenizer, batch_size, max_seq_length, split="train")

    total_diff = 0
    max_diff = 0
    num_mismatches = 0
    total_tokens = 0
    samples_processed = 0

    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    document_ids = batch['document_ids'].to(device)
    attention_mask = packed_document_causal_mask(document_ids).to(device)

    with torch.no_grad():
        masked_logits = reference_model(input_ids, attention_mask)

    comparison_results = compare_masked_and_individual_passes(
        reference_model, masked_logits, input_ids, document_ids, labels, device
    )

    for result in comparison_results:
        logger.info(f"Batch {result['batch'] + 1}, Sample {result['sample'] + 1}:")
        logger.info(f"  Chosen  - Max diff: {result['chosen_max_diff']:.6f}, Mean diff: {result['chosen_mean_diff']:.6f}, Mismatches: {result['chosen_mismatches']}")
        logger.info(f"  Rejected - Max diff: {result['rejected_max_diff']:.6f}, Mean diff: {result['rejected_mean_diff']:.6f}, Mismatches: {result['rejected_mismatches']}")

        total_diff += result['chosen_mean_diff'] + result['rejected_mean_diff']
        max_diff = max(max_diff, result['chosen_max_diff'], result['rejected_max_diff'])
        num_mismatches += result['chosen_mismatches'] + result['rejected_mismatches']
        total_tokens += result['total_tokens']
        samples_processed += 1

    avg_diff = total_diff / (2 * samples_processed)
    logger.info(f"Test completed. Average difference: {avg_diff:.6f}, Max difference: {max_diff:.6f}")
    logger.info(f"Total token prediction mismatches: {num_mismatches}")
    logger.info(f"Total tokens processed: {total_tokens}")
    dist.destroy_process_group()

    return avg_diff, max_diff, num_mismatches

def test_hh_mask_advanced():
    device, device_type = setup_environment()
    job_config = create_job_config()
    job_config.model.attention = "sdpa"
    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)
    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    reference_model = build_reference_model(job_config, world_mesh, parallel_dims, tokenizer)
    chat_format = ChatFormat(tokenizer)

    def encode_prompt(messages):
        tokens = [tokenizer.bos_id]
        for message in messages:
            toks, _ = chat_format.encode_message(message, tool_prompt_format=ToolPromptFormat.json)
            tokens.extend(toks)
        return tokens

    def encode_chosen_rejected(message):
        return chat_format.encode_dialog_prompt([message]).tokens[1:]

    # Define two sets of messages (unchanged)
    messages1 = [
        RawMessage(role="system", content="Reply with a number only"),
        RawMessage(role="user", content="Sum up all the numbers you see."),
        RawMessage(role="user", content="5")
    ]
    messages2 = [
        RawMessage(role="system", content="Reply with a number only"),
        RawMessage(role="user", content="Sum up all the numbers you see."),
        RawMessage(role="user", content="2")
    ]

    # Encode sequences (unchanged)
    prompt1 = encode_prompt(messages1)
    chosen1 = encode_chosen_rejected(RawMessage(role="user", content="12"))
    rejected1 = encode_chosen_rejected(RawMessage(role="user", content="10"))

    prompt2 = encode_prompt(messages2)
    chosen2 = encode_chosen_rejected(RawMessage(role="user", content="6"))
    rejected2 = encode_chosen_rejected(RawMessage(role="user", content="5"))

    # Create the packed sequence (unchanged)
    all_tokens = prompt1 + chosen1 + rejected1 + prompt2 + chosen2 + rejected2
    tokens = torch.tensor(all_tokens).unsqueeze(0).to(device)

    # Create option_ids and document_ids (unchanged)
    option_ids = torch.zeros_like(tokens, dtype=torch.long)
    option_ids[0, len(prompt1):len(prompt1) + len(chosen1)] = 1
    option_ids[0, len(prompt1) + len(chosen1):len(prompt1) + len(chosen1) + len(rejected1)] = 2
    option_ids[0, len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2):len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2) + len(chosen2)] = 1
    option_ids[0, len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2) + len(chosen2):] = 2

    document_ids = torch.zeros_like(tokens, dtype=torch.long)
    document_ids[0, len(prompt1) + len(chosen1) + len(rejected1):] = 1

    # Create the mask (unchanged)
    mask = packed_preference_causal_mask(option_ids, document_ids)

    # Masked forward pass (unchanged)
    with torch.no_grad():
        masked_logits = reference_model(tokens, mask)

    # Unmasked forward passes (unchanged)
    def unmasked_forward(sequence):
        with torch.no_grad():
            return reference_model(torch.tensor(sequence).unsqueeze(0).to(device), None)

    unmasked_logits1_chosen = unmasked_forward(prompt1 + chosen1)
    unmasked_logits1_rejected = unmasked_forward(prompt1 + rejected1)
    unmasked_logits2_chosen = unmasked_forward(prompt2 + chosen2)
    unmasked_logits2_rejected = unmasked_forward(prompt2 + rejected2)

    # New comparison logic using compare_logits_detailed
    logger.info("Comparing logits for the first prompt-chosen-rejected set:")
    compare_logits_detailed(
        masked_logits[0, :len(prompt1) + len(chosen1)],
        unmasked_logits1_chosen[0],
        "Prompt 1 + Chosen 1",
        tokenizer
    )
    compare_logits_detailed(
        torch.cat([masked_logits[0, :len(prompt1)],
                   masked_logits[0, len(prompt1) + len(chosen1):len(prompt1) + len(chosen1) + len(rejected1)]]),
        unmasked_logits1_rejected[0],
        "Prompt 1 + Rejected 1",
        tokenizer
    )

    logger.info("\nComparing logits for the second prompt-chosen-rejected set:")
    compare_logits_detailed(
        masked_logits[0,
        len(prompt1) + len(chosen1) + len(rejected1):len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2) + len(
            chosen2)],
        unmasked_logits2_chosen[0],
        "Prompt 2 + Chosen 2",
        tokenizer
    )
    compare_logits_detailed(
        torch.cat([
            masked_logits[0,
            len(prompt1) + len(chosen1) + len(rejected1):len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2)],
            masked_logits[0, len(prompt1) + len(chosen1) + len(rejected1) + len(prompt2) + len(chosen2):]
        ]),
        unmasked_logits2_rejected[0],
        "Prompt 2 + Rejected 2",
        tokenizer
    )

    # Sanity check: Compare second forward pass with itself
    unmasked_logits2_chosen_repeat = unmasked_forward(prompt2 + chosen2)
    compare_logits_detailed(unmasked_logits2_chosen[0], unmasked_logits2_chosen_repeat[0],
                            "Sanity Check: Second Forward Pass", tokenizer)

    dist.destroy_process_group()
    logger.info("Logit comparison completed. Check the detailed comparisons above.")

def test_hh_mask():
    device, device_type = setup_environment()
    job_config = create_job_config()

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, world_size=1, enable_loss_parallel=False)
    world_mesh = parallel_dims.build_mesh(device_type)

    tokenizer = build_tokenizer("llama", job_config.model.tokenizer_path)
    reference_model = build_reference_model(job_config, world_mesh, parallel_dims, tokenizer)
    chat_format = ChatFormat(tokenizer)

    prompt_messages = [
        RawMessage(role="system", content="Reply with a number only"),
        RawMessage(role="user", content="Sum up all the numbers you see."),
        RawMessage(role="user", content="5")
    ]

    prompt_tokens = []
    prompt_tokens.append(tokenizer.special_tokens["<|begin_of_text|>"])
    for message in prompt_messages:
        toks, _ = chat_format.encode_message(message, tool_prompt_format=ToolPromptFormat.json)
        prompt_tokens.extend(toks)

    chosen_message = RawMessage(role="user", content="7")
    chosen_tokens, _ = chat_format.encode_message(chosen_message, tool_prompt_format=ToolPromptFormat.json)

    rejected_message = RawMessage(role="user", content="15")
    rejected_tokens = chat_format.encode_dialog_prompt([rejected_message]).tokens
    rejected_tokens = rejected_tokens[1:]

    all_tokens = prompt_tokens + chosen_tokens + rejected_tokens
    tokens = torch.tensor(all_tokens).unsqueeze(0).to(device)

    option_ids = torch.zeros_like(tokens, dtype=torch.long)
    option_ids[0, len(prompt_tokens):len(prompt_tokens) + len(chosen_tokens)] = 1
    option_ids[0, len(prompt_tokens) + len(chosen_tokens):] = 2

    document_ids = torch.zeros_like(tokens, dtype=torch.long)

    mask = packed_preference_causal_mask(option_ids, document_ids)
    # mask = None

    with torch.no_grad():
        logits = reference_model(tokens, mask)

    final_logits = logits[0, -1]
    predicted_token = torch.argmax(final_logits).item()
    predicted_text = tokenizer.decode([predicted_token])

    logger.info(f"Predicted token: {predicted_token}")
    logger.info(f"Predicted text: {predicted_text}")

    expected_sum = sum(int(x) for x in "10 87 91 7".split() if x.isdigit())
    logger.info(f"Expected sum: {expected_sum}")
    logger.info(f"Predicted sum: {predicted_text}")

    if int(predicted_text) == expected_sum:
        logger.info("Block mask is working correctly!")
    else:
        logger.warning("Block mask may not be working as expected.")

    dist.destroy_process_group()

def test_hh_dataset():
    try:
        config = setup_config()
        tokenizer = build_tokenizer("llama", config.model.tokenizer_path)

        logger.info("Testing HH dataset with new HHDataset")
        dataset = HHDataset(
            tokenizer=tokenizer,
            split="train",
            seq_len=2048,
            cache_dir=None
        )

        # Get first batch
        batch_iter = iter(dataset)
        first_batch = next(batch_iter)

        logger.info(f"First batch keys: {first_batch.keys()}")
        logger.info(f"First batch shapes: tokens={first_batch['tokens'].shape}, "
                    f"labels={first_batch['labels'].shape}, "
                    f"option_ids={first_batch['option_ids'].shape}, "
                    f"input_pos={first_batch['input_pos'].shape}, "
                    f"document_ids={first_batch['document_ids'].shape}")

        # Decode first sequence
        decoded_text = tokenizer.decode(first_batch['tokens'].tolist())
        logger.info(f"First sequence preview: {decoded_text[:200]}...")

        logger.info("HH Dataset test completed successfully")

    except Exception as e:
        logger.error(f"Error in test_hh_dataset: {str(e)}", exc_info=True)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    init_logger()
    logger.info("Starting HH dataset tests")
    test_advanced_document_causal_mask(num_samples=50, max_seq_length=8192, batch_size=2)
    test_sdpa_mask_equivalence()
    test_hh_mask_advanced()
    test_hh_mask()
    test_hh_dataset()