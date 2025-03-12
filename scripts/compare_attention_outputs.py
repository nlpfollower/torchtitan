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


def compare_outputs(cp_outputs, dp_outputs, layer_idx, num_cp_ranks=2):
    """Compare CP and DP outputs for a given layer with chunk-based analysis"""
    if not cp_outputs or not dp_outputs:
        return {
            "layer_idx": layer_idx,
            "cp_outputs": len(cp_outputs),
            "dp_outputs": len(dp_outputs),
            "error": "Missing outputs"
        }

    # Get the first output from each (assuming only one per layer)
    assert len(cp_outputs) == 1, "Multiple CP outputs for layer"
    assert len(dp_outputs) == 1, "Multiple DP outputs for layer"
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

    # Calculate global differences
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

    # Chunk-based analysis for sequence dimension
    # Split sequence dimension (dim=2) into 2*num_cp_ranks chunks
    seq_chunks = 2 * num_cp_ranks
    seq_len = cp_output.shape[2]
    chunk_size = seq_len // seq_chunks

    chunk_metrics = []
    for i in range(seq_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < seq_chunks - 1 else seq_len

        # Extract chunks
        cp_chunk = cp_output[:, :, start_idx:end_idx, :]
        dp_chunk = dp_output[:, :, start_idx:end_idx, :]

        # Calculate differences for this chunk
        chunk_abs_diff = torch.abs(cp_chunk - dp_chunk)
        chunk_max_abs_diff = torch.max(chunk_abs_diff).item()
        chunk_mean_abs_diff = torch.mean(chunk_abs_diff).item()
        chunk_significant_diffs = (chunk_abs_diff > 1e-3).sum().item()
        chunk_total_elements = cp_chunk.numel()
        chunk_diff_percentage = 100 * chunk_significant_diffs / chunk_total_elements

        chunk_metrics.append({
            "chunk_idx": i,
            "seq_range": (start_idx, end_idx),
            "max_abs_diff": chunk_max_abs_diff,
            "mean_abs_diff": chunk_mean_abs_diff,
            "diff_percentage": chunk_diff_percentage
        })

    return {
        "layer_idx": layer_idx,
        "shape": cp_output.shape,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "significant_diffs": significant_diffs,
        "total_elements": total_elements,
        "diff_percentage": diff_percentage,
        "chunk_analysis": chunk_metrics
    }


def create_chunk_heatmap(results, output_path):
    """Create a heatmap visualization of differences by chunk and layer"""
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract data for heatmap
    layers = []
    chunk_indices = []
    diff_data = []

    for result in results:
        if "chunk_analysis" not in result:
            continue

        layer_idx = result["layer_idx"]
        for chunk in result["chunk_analysis"]:
            layers.append(layer_idx)
            chunk_indices.append(chunk["chunk_idx"])
            diff_data.append(chunk["diff_percentage"])

    if not layers:
        return

    # Convert to numpy arrays
    layers = np.array(layers)
    chunk_indices = np.array(chunk_indices)
    diff_data = np.array(diff_data)

    # Get unique layers and chunks
    unique_layers = np.sort(np.unique(layers))
    unique_chunks = np.sort(np.unique(chunk_indices))

    # Create heatmap data
    heatmap_data = np.zeros((len(unique_layers), len(unique_chunks)))

    for i, layer in enumerate(unique_layers):
        for j, chunk in enumerate(unique_chunks):
            mask = (layers == layer) & (chunk_indices == chunk)
            if np.any(mask):
                heatmap_data[i, j] = diff_data[mask][0]

    # Create heatmap visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Difference Percentage (%)')
    plt.title('Attention Output Differences by Layer and Chunk')
    plt.xlabel('Chunk Index')
    plt.ylabel('Layer Index')
    plt.xticks(np.arange(len(unique_chunks)), unique_chunks)
    plt.yticks(np.arange(len(unique_layers)), unique_layers)
    plt.tight_layout()
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser(description="Compare CP and DP attention outputs")
    parser.add_argument("--cp_trace", required=True, help="Path to CP attention trace file")
    parser.add_argument("--dp_trace", required=True, help="Path to DP attention trace file")
    parser.add_argument("--output", default="attention_comparison.txt", help="Output file")
    parser.add_argument("--num_cp_ranks", type=int, default=2, help="Number of CP ranks used")
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

        result = compare_outputs(cp_layer_outputs, dp_layer_outputs, layer_idx, args.num_cp_ranks)
        results.append(result)

        # Print layer result
        print(f"Layer {layer_idx}:")
        for k, v in result.items():
            if k != "layer_idx" and k != "chunk_analysis":
                print(f"  {k}: {v}")

        # Print chunk analysis summary
        if "chunk_analysis" in result:
            print(f"  Chunk Analysis:")
            for chunk in result["chunk_analysis"]:
                print(
                    f"    Chunk {chunk['chunk_idx']} ({chunk['seq_range']}): max_diff={chunk['max_abs_diff']:.6f}, mean_diff={chunk['mean_abs_diff']:.6f}, diff_pct={chunk['diff_percentage']:.2f}%")
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
                if k != "layer_idx" and k != "chunk_analysis":
                    f.write(f"  {k}: {v}\n")

            # Write chunk analysis
            if "chunk_analysis" in result:
                f.write("  Chunk Analysis:\n")
                for chunk in result["chunk_analysis"]:
                    f.write(f"    Chunk {chunk['chunk_idx']} ({chunk['seq_range']}): ")
                    f.write(f"max_diff={chunk['max_abs_diff']:.6f}, ")
                    f.write(f"mean_diff={chunk['mean_abs_diff']:.6f}, ")
                    f.write(f"diff_pct={chunk['diff_percentage']:.2f}%\n")
            f.write("\n")

    print(f"Detailed results written to {args.output}")

    # Create visualization of differences by chunk
    # This will help us see if certain parts of the sequence are more affected
    create_chunk_heatmap(results, args.output.replace(".txt", "_chunks.png"))
    print(f"Chunk analysis visualization saved to {args.output.replace('.txt', '_chunks.png')}")

    # Create overall layer difference visualization
    plt.figure(figsize=(10, 6))
    layer_nums = [r["layer_idx"] for r in results if "diff_percentage" in r]
    diff_pcts = [r["diff_percentage"] for r in results if "diff_percentage" in r]

    if layer_nums and diff_pcts:
        plt.bar(layer_nums, diff_pcts)
        plt.title("Percentage of Different Elements by Layer")
        plt.xlabel("Layer")
        plt.ylabel("% Different Elements")
        plt.savefig(args.output.replace(".txt", ".png"))
        print(f"Overall visualization saved to {args.output.replace('.txt', '.png')}")


if __name__ == "__main__":
    main()