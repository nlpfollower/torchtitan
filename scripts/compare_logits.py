#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare logits between two files")
    parser.add_argument("--file1", type=str, required=True,
                        help="Path to first logits file (reference)")
    parser.add_argument("--file2", type=str, required=True,
                        help="Path to second logits file (test)")
    parser.add_argument("--common_dtype", type=str, default="float32",
                        help="Common dtype to convert tensors to before comparison (float32, float64)")
    return parser.parse_args()


def load_logits(filepath):
    """Load logits and metadata from a saved file"""
    try:
        data = torch.load(filepath, map_location=torch.device('cpu'))
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None


def get_dtype_str(tensor):
    """Get a readable string representation of tensor dtype"""
    return str(tensor.dtype).split('.')[-1]


def convert_to_common_dtype(tensor1, tensor2, dtype_str="float32"):
    """Convert both tensors to a common dtype for comparison"""
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64
    }

    common_dtype = dtype_map.get(dtype_str, torch.float32)

    # Convert both tensors to the common dtype
    tensor1_converted = tensor1.to(common_dtype)
    tensor2_converted = tensor2.to(common_dtype)

    return tensor1_converted, tensor2_converted


def analyze_differences(ref_logits, test_logits):
    """Calculate basic statistics on the differences between two logit tensors"""
    # Flatten tensors for absolute difference analysis
    ref_flat = ref_logits.flatten()
    test_flat = test_logits.flatten()

    # Calculate absolute differences
    abs_diff = torch.abs(ref_flat - test_flat)

    # Calculate basic stats
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()

    # Find positions of largest differences
    max_diff_idx = torch.argmax(abs_diff).item()
    orig_shape = ref_logits.shape

    # Convert flat index to multi-dimensional index
    if len(orig_shape) > 1:
        multi_indices = np.unravel_index(max_diff_idx, orig_shape)
        pos_str = f"position {multi_indices}"
    else:
        pos_str = f"index {max_diff_idx}"

    # Get values at max difference position
    ref_val = ref_flat[max_diff_idx].item()
    test_val = test_flat[max_diff_idx].item()

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_diff_pos": pos_str,
        "max_diff_ref_value": ref_val,
        "max_diff_test_value": test_val
    }


def analyze_argmax_differences(ref_logits, test_logits):
    """Analyze differences in argmax predictions between two logit tensors"""
    # Expected shape: [batch_size, seq_len, vocab_size]
    batch_size, seq_len, vocab_size = ref_logits.shape

    # Compute argmax along the vocabulary dimension (dim=2)
    ref_argmax = torch.argmax(ref_logits, dim=2)  # Shape: [batch_size, seq_len]
    test_argmax = torch.argmax(test_logits, dim=2)  # Shape: [batch_size, seq_len]

    # Count positions where argmax differs
    argmax_diff_mask = (ref_argmax != test_argmax)  # Shape: [batch_size, seq_len]
    total_argmax_diffs = torch.sum(argmax_diff_mask).item()
    total_positions = batch_size * seq_len
    argmax_diff_percentage = (total_argmax_diffs / total_positions) * 100

    # Find some example differences for inspection
    # Get indices where differences occur (up to 5 examples)
    diff_indices = argmax_diff_mask.nonzero(as_tuple=False)
    num_examples = min(5, diff_indices.size(0))
    examples = []

    for i in range(num_examples):
        b, s = diff_indices[i].tolist()
        ref_token = ref_argmax[b, s].item()
        test_token = test_argmax[b, s].item()
        examples.append({
            "batch": b,
            "seq_pos": s,
            "ref_token_id": ref_token,
            "test_token_id": test_token
        })

    return {
        "total_argmax_diffs": total_argmax_diffs,
        "total_positions": total_positions,
        "argmax_diff_percentage": argmax_diff_percentage,
        "example_diffs": examples
    }


def get_parallelism_config_str(metadata):
    """Create a string describing the parallelism configuration"""
    if 'parallel_config' in metadata:
        config = metadata['parallel_config']
        return f"DP{config['dp_shard']}_CP{config['cp']}_TP{config['tp']}_PP{config['pp']}"
    else:
        return "Unknown"


def main():
    args = parse_args()

    # Load both files
    file1_data = load_logits(args.file1)
    file2_data = load_logits(args.file2)

    if file1_data is None or file2_data is None:
        print("Error loading one or both files. Exiting.")
        return

    # Extract logits and metadata
    file1_logits = file1_data["logits"]
    file2_logits = file2_data["logits"]

    # Print original dtypes
    file1_dtype = get_dtype_str(file1_logits)
    file2_dtype = get_dtype_str(file2_logits)

    file1_metadata = file1_data.get("metadata", {})
    file2_metadata = file2_data.get("metadata", {})

    file1_config = get_parallelism_config_str(file1_metadata)
    file2_config = get_parallelism_config_str(file2_metadata)

    print(f"File 1: {args.file1}")
    print(f"  Config: {file1_config}")
    print(f"  Shape: {file1_logits.shape}")
    print(f"  Dtype: {file1_dtype}")

    print(f"File 2: {args.file2}")
    print(f"  Config: {file2_config}")
    print(f"  Shape: {file2_logits.shape}")
    print(f"  Dtype: {file2_dtype}")

    # Verify tensor shapes match
    if file1_logits.shape != file2_logits.shape:
        print(f"ERROR: Shape mismatch: File 1 {file1_logits.shape} vs File 2 {file2_logits.shape}")
        return

    # Convert to common dtype for comparison
    print(f"Converting both tensors to {args.common_dtype} for comparison...")
    file1_logits_converted, file2_logits_converted = convert_to_common_dtype(
        file1_logits, file2_logits, args.common_dtype
    )

    # Basic difference analysis
    diff_stats = analyze_differences(file1_logits_converted, file2_logits_converted)

    # Argmax difference analysis
    argmax_stats = analyze_argmax_differences(file1_logits_converted, file2_logits_converted)

    # Print results
    print("\n=== Basic Difference Analysis ===")
    print(f"Max absolute difference: {diff_stats['max_abs_diff']:.6e} at {diff_stats['max_diff_pos']}")
    print(f"  - File 1 value: {diff_stats['max_diff_ref_value']:.6e}")
    print(f"  - File 2 value: {diff_stats['max_diff_test_value']:.6e}")
    print(f"Mean absolute difference: {diff_stats['mean_abs_diff']:.6e}")

    print("\n=== Argmax Difference Analysis ===")
    print(
        f"Total positions with different argmax: {argmax_stats['total_argmax_diffs']} out of {argmax_stats['total_positions']}")
    print(f"Percentage of positions with different argmax: {argmax_stats['argmax_diff_percentage']:.4f}%")

    if argmax_stats['example_diffs']:
        print("\nExample differences (batch, seq_pos, ref_token_id, test_token_id):")
        for i, example in enumerate(argmax_stats['example_diffs']):
            print(
                f"  {i + 1}. ({example['batch']}, {example['seq_pos']}): {example['ref_token_id']} vs {example['test_token_id']}")

    # Print timing information if available
    if 'elapsed_time' in file1_metadata and 'elapsed_time' in file2_metadata:
        time1 = file1_metadata['elapsed_time']
        time2 = file2_metadata['elapsed_time']
        speedup = time1 / time2 if time2 > 0 else 0

        print("\n=== Performance Comparison ===")
        print(f"File 1 execution time: {time1:.3f}s")
        print(f"File 2 execution time: {time2:.3f}s")
        print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()