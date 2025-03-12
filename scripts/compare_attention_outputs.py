#!/usr/bin/env python3
"""
Compare attention outputs between CP and DP runs.
"""

import os
import sys
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

import torch


def load_attention_outputs(filepath):
    """Load attention outputs from pickle file"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_layer_outputs(attention_outputs, layer_idx, parallel_type):
    """Extract outputs for a specific layer and parallel type"""
    if parallel_type == "cp":
        # Use merged results for CP
        return [out["output"] for out in attention_outputs["cp_merged"]
                if out["layer_idx"] == layer_idx]
    else:
        # Use DP outputs
        return [out["output"] for out in attention_outputs["dp"]
                if out["layer_idx"] == layer_idx]


def compare_outputs(cp_outputs, dp_outputs, layer_idx):
    """Compare CP and DP outputs for a given layer"""
    if not cp_outputs or not dp_outputs:
        return {
            "layer_idx": layer_idx,
            "cp_outputs": len(cp_outputs),
            "dp_outputs": len(dp_outputs),
            "error": "Missing outputs"
        }

    # Get the first output from each (assuming only one per layer)
    cp_output = cp_outputs[0]
    dp_output = dp_outputs[0]

    # Check shapes
    if cp_output.shape != dp_output.shape:
        return {
            "layer_idx": layer_idx,
            "cp_shape": cp_output.shape,
            "dp_shape": dp_output.shape,
            "error": "Shape mismatch"
        }

    # Calculate differences
    abs_diff = torch.abs(cp_output - dp_output)
    rel_diff = abs_diff / (torch.abs(dp_output) + 1e-8)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Count significant differences
    significant_diffs = (abs_diff > 1e-3).sum().item()
    total_elements = cp_output.numel()
    diff_percentage = 100 * significant_diffs / total_elements

    return {
        "layer_idx": layer_idx,
        "shape": cp_output.shape,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "significant_diffs": significant_diffs,
        "total_elements": total_elements,
        "diff_percentage": diff_percentage
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CP and DP attention outputs")
    parser.add_argument("--cp_trace", required=True, help="Path to CP attention trace file")
    parser.add_argument("--dp_trace", required=True, help="Path to DP attention trace file")
    parser.add_argument("--output", default="attention_comparison.txt", help="Output file")
    args = parser.parse_args()

    import torch  # Import here to avoid requirement when not needed

    # Load outputs
    cp_outputs = load_attention_outputs(args.cp_trace)
    dp_outputs = load_attention_outputs(args.dp_trace)

    print(f"Loaded CP trace with {len(cp_outputs['cp_merged'])} merged outputs")
    print(f"Loaded DP trace with {len(dp_outputs['dp'])} outputs")

    # Get all layer indices
    layer_indices = set()
    for out in cp_outputs["cp_merged"]:
        layer_indices.add(out["layer_idx"])
    for out in dp_outputs["dp"]:
        layer_indices.add(out["layer_idx"])
    layer_indices = sorted(layer_indices)

    # Compare each layer
    results = []
    for layer_idx in layer_indices:
        cp_layer_outputs = get_layer_outputs(cp_outputs, layer_idx, "cp")
        dp_layer_outputs = get_layer_outputs(dp_outputs, layer_idx, "dp")

        result = compare_outputs(cp_layer_outputs, dp_layer_outputs, layer_idx)
        results.append(result)

        # Print layer result
        print(f"Layer {layer_idx}:")
        for k, v in result.items():
            if k != "layer_idx":
                print(f"  {k}: {v}")
        print()

    # Find the first layer with significant differences
    first_diff_layer = None
    for result in results:
        if "diff_percentage" in result and result["diff_percentage"] > 1.0:
            first_diff_layer = result["layer_idx"]
            break

    if first_diff_layer is not None:
        print(f"First layer with significant differences: {first_diff_layer}")

    # Write detailed results to file
    with open(args.output, "w") as f:
        f.write("Attention Output Comparison Results\n")
        f.write("=================================\n\n")

        if first_diff_layer is not None:
            f.write(f"First layer with significant differences: {first_diff_layer}\n\n")

        for result in results:
            f.write(f"Layer {result['layer_idx']}:\n")
            for k, v in result.items():
                if k != "layer_idx":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"Detailed results written to {args.output}")

    # Create visualization of differences
    plt.figure(figsize=(10, 6))
    layer_nums = [r["layer_idx"] for r in results if "diff_percentage" in r]
    diff_pcts = [r["diff_percentage"] for r in results if "diff_percentage" in r]

    if layer_nums and diff_pcts:
        plt.bar(layer_nums, diff_pcts)
        plt.title("Percentage of Different Elements by Layer")
        plt.xlabel("Layer")
        plt.ylabel("% Different Elements")
        plt.yscale("log")
        plt.savefig(args.output.replace(".txt", ".png"))
        print(f"Visualization saved to {args.output.replace('.txt', '.png')}")


if __name__ == "__main__":
    main()