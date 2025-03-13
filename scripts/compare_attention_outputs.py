#!/usr/bin/env python3
"""
Compare attention outputs between CP and DP runs.
"""
import math
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
        cp_outputs = []
        for out in attention_outputs["cp_merged"]:
            if out["layer_idx"] == layer_idx:
                # Use unsharded output if available, otherwise use regular output
                if "unsharded_output" in out and out["unsharded_output"] is not None:
                    cp_outputs.append(out["unsharded_output"])
                else:
                    cp_outputs.append(out["output"])
        return cp_outputs
    else:
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


def visualize_mask_pair(
        cp_mask,
        dp_mask,
        dp_q_range,
        dp_k_range,
        save_path,
        title="Mask Comparison",
        max_vis_size=2048
):
    """
    Visualizes CP and DP masks side by side with their difference.

    Args:
        cp_mask: Context parallel mask tensor
        dp_mask: Data parallel mask tensor
        dp_q_range: Tuple of (start, end) for query slice in DP mask
        dp_k_range: Tuple of (start, end) for key slice in DP mask
        save_path: Where to save the plot
        title: Title for the plot
        max_vis_size: Maximum dimension size before downsampling
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Handle mask being None
    if cp_mask is None:
        print(f"Cannot visualize: CP mask is None for {title}")
        return

    if dp_mask is None:
        print(f"Cannot visualize: DP mask is None for {title}")
        return

    # Extract the relevant slice from DP mask
    q_start, q_end = dp_q_range
    k_start, k_end = dp_k_range

    # Extract DP submask for the specified region
    if dp_mask.dim() >= 4:
        dp_submask = dp_mask[0, 0, q_start:q_end, k_start:k_end]
    else:
        dp_submask = dp_mask[q_start:q_end, k_start:k_end]

    # Extract corresponding part of CP mask
    if cp_mask.dim() >= 4:
        cp_mask_slice = cp_mask[0, 0]
    elif cp_mask.dim() == 3:
        cp_mask_slice = cp_mask[0]
    else:
        cp_mask_slice = cp_mask  # Already 2D

    # Convert to float for comparison
    dp_submask = dp_submask.float()
    cp_mask_slice = cp_mask_slice.float()

    # Check if masks need downsampling (too large to visualize)
    needs_downsampling = False

    if max(cp_mask_slice.shape) > max_vis_size or max(dp_submask.shape) > max_vis_size:
        needs_downsampling = True

        # Calculate downsampling factor based on the larger mask
        max_dim = max(max(cp_mask_slice.shape), max(dp_submask.shape))
        ds_factor = max(1, max_dim // max_vis_size)

        # Downsample masks
        if cp_mask_slice.dim() == 2:
            cp_ds = cp_mask_slice[::ds_factor, ::ds_factor]
            cp_mask_slice = cp_ds

        if dp_submask.dim() == 2:
            dp_ds = dp_submask[::ds_factor, ::ds_factor]
            dp_submask = dp_ds

        print(f"Downsampled masks for visualization (factor: {ds_factor})")

    # Convert to numpy for plotting
    cp_np = cp_mask_slice.cpu().numpy()
    dp_np = dp_submask.cpu().numpy()

    # Calculate difference
    # If shapes don't match, we'll just show the masks separately without difference
    same_shape = cp_np.shape == dp_np.shape
    if same_shape:
        diff = cp_np - dp_np

    # Create figure
    n_plots = 3 if same_shape else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    # Determine colormap and range
    if cp_mask.dtype == torch.bool or dp_mask.dtype == torch.bool:
        # For boolean masks: red (masked/False) to blue (attended/True)
        colors = ['#ffcccc', '#99ccff']
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=2)
        vmin, vmax = 0, 1
    else:
        # For float masks: use Blues colormap
        cmap = 'Blues'
        # Determine bounds from data
        vmin = min(float(cp_np.min()), float(dp_np.min()))
        vmax = max(float(cp_np.max()), float(dp_np.max()))

        # If all values are the same, add a small range
        if vmin == vmax:
            vmin = vmin - 0.1 if vmin > 0 else 0
            vmax = vmax + 0.1 if vmax < 1 else 1

    # Diff colormap
    diff_cmap = 'RdBu_r'

    # Plot CP mask
    im1 = axes[0].imshow(cp_np, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    axes[0].set_title(f"CP Mask\nShape: {tuple(cp_mask.shape)}")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im1, ax=axes[0])

    # Plot DP mask
    im2 = axes[1].imshow(dp_np, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    axes[1].set_title(f"DP Mask\nQ[{q_start}:{q_end}], K[{k_start}:{k_end}]")
    axes[1].set_xlabel("Key position")
    plt.colorbar(im2, ax=axes[1])

    # Plot difference if shapes match
    if same_shape:
        im3 = axes[2].imshow(diff, cmap=diff_cmap, vmin=-1, vmax=1, interpolation='none')
        axes[2].set_title("Difference (CP - DP)")
        axes[2].set_xlabel("Key position")
        plt.colorbar(im3, ax=axes[2])

    # Add grid lines if masks are small enough
    for i, ax in enumerate(axes):
        if i < 2 or same_shape:  # Skip if we don't have a diff plot
            curr_arr = cp_np if i == 0 else dp_np if i == 1 else diff
            h, w = curr_arr.shape

            if max(h, w) <= 128:
                # Add major grid lines at section boundaries
                grid_interval = max(1, min(h, w) // 4)
                for j in range(grid_interval, max(h, w), grid_interval):
                    if j < h:
                        ax.axhline(y=j - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                    if j < w:
                        ax.axvline(x=j - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    # Add main title
    plt.suptitle(f"{title}\n{'(Downsampled)' if needs_downsampling else ''}")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved mask comparison to {save_path}")
    plt.close(fig)


def analyze_q_k_mask_chunks(cp_outputs, dp_outputs, num_cp_ranks, rank=0, output_dir="./visualizations"):
    """Analyze how query, key, and mask chunks are processed in CP vs DP, focusing on layer 0"""
    if "cp_detailed" not in cp_outputs:
        print(f"\nNo detailed tensor data found for rank {rank}. Please run with enhanced tracing.")
        return

    cp_details = cp_outputs.get("cp_detailed", [])
    dp_details = dp_outputs.get("dp_detailed", [])

    # Focus only on layer 0
    target_layer = 0

    # Get details for this layer
    cp_layer = [d for d in cp_details if d.get("layer_idx", -1) == target_layer]
    dp_layer = [d for d in dp_details if d.get("layer_idx", -1) == target_layer]

    if not cp_layer:
        print(f"No data found for layer {target_layer} on rank {rank}")
        return

    if not dp_layer:
        print("No DP data found for comparison")
        return

    # DP data - just take the first item for this layer
    dp_data = dp_layer[0]
    dp_q = dp_data.get("q")
    dp_k = dp_data.get("k")
    dp_mask = dp_data.get("mask")

    if dp_q is None or dp_k is None:
        print("Missing query/key data in DP trace")
        return

    print(f"\n===== Analyzing Layer {target_layer} CP Chunking (Rank {rank}) =====")

    # Calculate sequence lengths and chunk sizes
    seq_len = dp_q.size(2)
    chunk_size = seq_len // (2 * num_cp_ranks)

    # Determine processing pattern from round robin algorithm
    rank_process_info = {}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each CP operation to analyze mask differences
    for detail in sorted(cp_layer, key=lambda x: x.get("iteration", 0)):
        iter_num = detail.get("iteration", 0)
        cp_q = detail.get("q")
        cp_k = detail.get("k")
        cp_mask = detail.get("mask")

        if cp_q is None or cp_k is None:
            continue

        # Calculate source rank for this iteration (for key chunks)
        source_rank = (rank - iter_num) % num_cp_ranks

        print(f"\nIteration {iter_num} - Source Rank {source_rank}:")
        print(f"  Q shape: {tuple(cp_q.shape)}, K shape: {tuple(cp_k.shape)}")

        # Store the processing pattern for this iteration
        if rank not in rank_process_info:
            rank_process_info[rank] = []

        # Determine query and key chunks for this iteration
        # In round-robin, each rank r gets chunks [r, 2*size-1-r]
        q_chunks = []
        k_chunks = []

        if iter_num == 0:
            # First iteration: process both local chunks against both chunks from same rank
            q_chunks = [rank, 2 * num_cp_ranks - 1 - rank]
            k_chunks = [source_rank, 2 * num_cp_ranks - 1 - source_rank]
        elif iter_num <= rank:
            # Process all queries against first chunk of keys from source_rank
            q_chunks = [rank, 2 * num_cp_ranks - 1 - rank]
            k_chunks = [source_rank]
        else:
            # Process second half of queries against all chunks of keys from source_rank
            q_chunks = [2 * num_cp_ranks - 1 - rank]
            k_chunks = [source_rank, 2 * num_cp_ranks - 1 - source_rank]

        rank_process_info[rank].append({
            "iter": iter_num,
            "source_rank": source_rank,
            "q_chunks": q_chunks,
            "k_chunks": k_chunks,
            "q_shape": cp_q.shape,
            "k_shape": cp_k.shape
        })

        # Compare masks if available
        if cp_mask is not None and dp_mask is not None:
            print("  Mask regions and DP comparison:")

            # Calculate global start/end positions for query and key chunks
            for q_idx in q_chunks:
                q_start = q_idx * chunk_size
                q_end = (q_idx + 1) * chunk_size

                for k_idx in k_chunks:
                    k_start = k_idx * chunk_size
                    k_end = (k_idx + 1) * chunk_size

                    print(f"    Region Q[{q_start}:{q_end}], K[{k_start}:{k_end}]:")

                    # Extract corresponding region from DP mask
                    if dp_mask.dim() >= 4:
                        dp_submask = dp_mask[0, 0, q_start:q_end, k_start:k_end]
                    else:
                        dp_submask = dp_mask[q_start:q_end, k_start:k_end]

                    # Compute mask statistics
                    if cp_mask.dim() >= 4:
                        cp_mask_float = cp_mask[0, 0].float()
                    else:
                        cp_mask_float = cp_mask.float()

                    if dp_submask.dtype == torch.bool:
                        dp_mask_float = dp_submask.float()
                    else:
                        dp_mask_float = dp_submask

                    # Count non-zero positions
                    cp_values = cp_mask_float.sum().item()
                    dp_values = dp_mask_float.sum().item()

                    # Calculate ratio (avoid division by zero)
                    mask_ratio = cp_values / max(dp_values, 1)

                    print(f"      CP Mask has {cp_values} non-zero positions")
                    print(f"      DP Mask has {dp_values} non-zero positions in this region")
                    print(f"      Ratio CP/DP: {mask_ratio:.2f}")

                    # Create side-by-side visualization of CP mask and relevant DP mask
                    comparison_path = os.path.join(
                        output_dir,
                        f"mask_comparison_layer{target_layer}_rank{rank}_iter{iter_num}_q{q_idx}_k{k_idx}.png"
                    )

                    # Use the new visualization function
                    visualize_mask_pair(
                        cp_mask=cp_mask,
                        dp_mask=dp_mask,
                        dp_q_range=(q_start, q_end),
                        dp_k_range=(k_start, k_end),
                        save_path=comparison_path,
                        title=f"Layer {target_layer}, Rank {rank}, Iter {iter_num}: Q{q_idx}, K{k_idx}"
                    )

    # Print the chunk processing assignment table
    print_chunk_assignment_table(rank_process_info.get(rank, []), num_cp_ranks, rank)

def print_chunk_assignment_table(process_info, num_cp_ranks, rank):
    """Print a visualization of chunk assignments for a specific rank"""
    # Create grid for tracking chunk assignments
    grid = np.full((2 * num_cp_ranks, 2 * num_cp_ranks), None)

    # Fill grid with processing information
    for info in process_info:
        iter_num = info.get("iter", -1)
        q_chunks = info.get("q_chunks", [])
        k_chunks = info.get("k_chunks", [])

        for q_idx in q_chunks:
            for k_idx in k_chunks:
                if 0 <= q_idx < 2 * num_cp_ranks and 0 <= k_idx < 2 * num_cp_ranks:
                    grid[q_idx][k_idx] = (rank, iter_num)

    # Print the grid
    print("\nChunk Processing Assignment:")
    print("  ", end="")
    for k in range(2 * num_cp_ranks):
        print(f"k{k:<2}", end=" ")
    print()

    for q in range(2 * num_cp_ranks):
        print(f"q{q:<2}", end=" ")
        for k in range(2 * num_cp_ranks):
            cell = grid[q][k]
            if cell:
                r, i = cell
                print(f"R{r}I{i}", end=" ")
            else:
                print("----", end=" ")
        print()

def visualize_attention_mask(
        mask,
        batch_idx=0,
        head_idx=0,
        save_path=None,
        title="Attention Mask"
):
    """
    Visualizes a boolean attention mask.
    Args:
        mask: Tensor of shape (batch_size, heads, seq_len, seq_len) or 2D mask
        batch_idx: Which batch to visualize
        head_idx: Which attention head to visualize
        save_path: Where to save the plot. If None, displays it
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Handle different mask shapes
    if mask.dim() >= 4:
        mask_slice = mask[batch_idx, head_idx].cpu().float()  # Shape: (seq_len, seq_len)
    elif mask.dim() == 3:
        mask_slice = mask[batch_idx].cpu().float()  # Shape: (seq_len, seq_len)
    else:
        mask_slice = mask.cpu().float()  # Already 2D

    # Ensure mask is 2D
    if mask_slice.dim() != 2:
        print(f"Warning: Mask has unexpected shape: {mask_slice.shape}")
        if mask_slice.numel() == 0:
            print("Cannot visualize empty mask")
            return
        # Try to reshape
        seq_len = int(math.sqrt(mask_slice.numel()))
        if seq_len * seq_len == mask_slice.numel():
            mask_slice = mask_slice.reshape(seq_len, seq_len)
        else:
            print("Cannot reshape mask to 2D")
            return

    seq_len = mask_slice.shape[0]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom two-color scheme: red (masked) to blue (attended)
    colors = ['#ffcccc', '#99ccff']
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=2)

    # Plot the mask
    im = ax.imshow(mask_slice, cmap=cmap, interpolation='none')
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Add colorbar with clear labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Masked (0)', 'Attended (1)'])

    # Add chunking lines if sequence length is large
    chunk_size = seq_len // 4  # Assume 4 chunks
    for i in range(1, 4):
        plt.axhline(y=i * chunk_size - 0.5, color='r', linestyle='-', linewidth=1)
        plt.axvline(x=i * chunk_size - 0.5, color='r', linestyle='-', linewidth=1)

    # Add position labels if sequence length is manageable
    if seq_len <= 32:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels([f'K{i}' for i in range(seq_len)])
        ax.set_yticklabels([f'Q{i}' for i in range(seq_len)])
        ax.tick_params(axis='both', which='major', labelsize=8)

        # Add grid
        ax.set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_mask_comparison(cp_mask, dp_mask, save_path=None, title="Mask Comparison"):
    """
    Visualize the CP and DP masks side by side with difference.

    Args:
        cp_mask: Context parallel mask tensor
        dp_mask: Data parallel mask tensor
        save_path: Path to save the visualization
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure both masks are 2D
    if cp_mask.dim() > 2:
        cp_mask = cp_mask.reshape(cp_mask.shape[-2], cp_mask.shape[-1])
    if dp_mask.dim() > 2:
        dp_mask = dp_mask.reshape(dp_mask.shape[-2], dp_mask.shape[-1])

    # Convert to numpy for visualization
    cp_np = cp_mask.float().cpu().numpy()
    dp_np = dp_mask.float().cpu().numpy()

    # Ensure shapes match for comparison, pad if necessary
    if cp_np.shape != dp_np.shape:
        max_rows = max(cp_np.shape[0], dp_np.shape[0])
        max_cols = max(cp_np.shape[1], dp_np.shape[1])

        cp_padded = np.zeros((max_rows, max_cols))
        dp_padded = np.zeros((max_rows, max_cols))

        cp_padded[:cp_np.shape[0], :cp_np.shape[1]] = cp_np
        dp_padded[:dp_np.shape[0], :dp_np.shape[1]] = dp_np

        cp_np = cp_padded
        dp_np = dp_padded

    # Calculate difference
    diff = cp_np - dp_np

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot CP mask
    im1 = ax1.imshow(cp_np, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title("CP Mask")
    ax1.set_xlabel("Key position")
    ax1.set_ylabel("Query position")
    plt.colorbar(im1, ax=ax1)

    # Plot DP mask
    im2 = ax2.imshow(dp_np, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title("DP Mask")
    ax2.set_xlabel("Key position")
    plt.colorbar(im2, ax=ax2)

    # Plot difference
    im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    ax3.set_title("Difference (CP - DP)")
    ax3.set_xlabel("Key position")
    plt.colorbar(im3, ax=ax3)

    # Add chunk boundary lines
    rows, cols = cp_np.shape
    chunk_size_rows = rows // 4
    chunk_size_cols = cols // 4

    for ax in [ax1, ax2, ax3]:
        # Horizontal lines (query chunks)
        for i in range(1, 4):
            ax.axhline(i * chunk_size_rows - 0.5, color='black', linestyle='-', linewidth=0.5)

        # Vertical lines (key chunks)
        for i in range(1, 4):
            ax.axvline(i * chunk_size_cols - 0.5, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison visualization saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def reconstruct_cp_mask(cp_outputs, dp_mask, num_cp_ranks):
    """
    Attempt to reconstruct the full CP mask from trace data

    Args:
        cp_outputs: The CP trace outputs
        dp_mask: The original mask from DP (for shape reference)
        num_cp_ranks: Number of CP ranks used

    Returns:
        Reconstructed mask tensor
    """
    # Create an empty mask with the same shape as dp_mask
    reconstructed = torch.zeros_like(dp_mask)

    # Extract detailed CP mask data if available
    if "cp_detailed" not in cp_outputs:
        print("No detailed mask information available in CP trace")
        return reconstructed

    # Get sequence length and chunk size
    seq_len = dp_mask.size(2) if dp_mask.dim() >= 3 else dp_mask.size(0)
    chunk_size = seq_len // (2 * num_cp_ranks)

    # Group by layer to find mask information
    layer_data = {}
    for item in cp_outputs["cp_detailed"]:
        layer_idx = item.get("layer_idx", -1)
        if layer_idx not in layer_data:
            layer_data[layer_idx] = []
        layer_data[layer_idx].append(item)

    # Use data from all layers to better reconstruct
    for layer_idx, items in layer_data.items():
        for item in items:
            rank = item.get("rank", -1)
            iter_num = item.get("iteration", -1)
            mask = item.get("mask")

            if mask is None:
                continue

            # Calculate source rank
            source_rank = (rank - iter_num) % num_cp_ranks

            # Calculate query and key chunks for this operation
            if iter_num == 0:
                # First iteration: both chunks for this rank
                q_chunks = [rank, 2 * num_cp_ranks - 1 - rank]
                k_chunks = [source_rank, 2 * num_cp_ranks - 1 - source_rank]
            elif iter_num <= rank:
                # Use first half of queries with keys from source rank
                q_chunks = [rank, 2 * num_cp_ranks - 1 - rank]
                k_chunks = [source_rank]
            else:
                # Use second half of queries with keys from source rank
                q_chunks = [2 * num_cp_ranks - 1 - rank]
                k_chunks = [source_rank, 2 * num_cp_ranks - 1 - source_rank]

            # Mark these chunks in the reconstructed mask
            try:
                for q_idx in q_chunks:
                    q_start = q_idx * chunk_size
                    q_end = (q_idx + 1) * chunk_size

                    for k_idx in k_chunks:
                        k_start = k_idx * chunk_size
                        k_end = (k_idx + 1) * chunk_size

                        # Set this region to 1 in the reconstructed mask
                        if reconstructed.dim() >= 4:
                            reconstructed[:, :, q_start:q_end, k_start:k_end] = 1.0
                        else:
                            reconstructed[q_start:q_end, k_start:k_end] = 1.0
            except Exception as e:
                print(f"Error reconstructing mask: {e}")

    return reconstructed

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


def visualize_raw_mask(
        mask,
        save_path,
        title="Attention Mask",
        max_vis_size=2048
):
    """
    Visualizes a mask tensor exactly as provided, with efficient handling for large masks.

    Args:
        mask: Tensor of any shape containing mask values
        save_path: Where to save the plot
        title: Title for the plot
        max_vis_size: Maximum dimension size before downsampling
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import math

    # Handle mask being None
    if mask is None:
        print(f"Cannot visualize: mask is None for {title}")
        return

    # First handle common mask shapes
    if mask.dim() >= 4:
        # For 4D masks (batch, heads, seq, seq), take first batch and head
        mask_to_vis = mask[0, 0]
    elif mask.dim() == 3:
        # For 3D masks (batch, seq, seq), take first batch
        mask_to_vis = mask[0]
    else:
        # Already 2D or 1D
        mask_to_vis = mask

    # Convert to 2D if needed - for non-square masks, visualize as is
    if mask_to_vis.dim() > 2:
        print(f"Warning: Unusual mask shape: {mask_to_vis.shape} for {title}")
        # Try to flatten to 2D by combining dimensions if possible
        if mask_to_vis.dim() == 3 and mask_to_vis.size(0) == 1:
            mask_to_vis = mask_to_vis.squeeze(0)
        else:
            # Just take the first slice if we can't make sense of it
            mask_to_vis = mask_to_vis[0]

    # Convert to float for visualization
    mask_to_vis = mask_to_vis.float().cpu()

    # Check if mask needs downsampling (too large to visualize)
    needs_downsampling = False
    if mask_to_vis.dim() == 2:
        h, w = mask_to_vis.shape
        if h > max_vis_size or w > max_vis_size:
            needs_downsampling = True

            # Calculate downsampling factor
            ds_factor = max(1, max(h, w) // max_vis_size)

            # Downsample using strided selection for large masks
            mask_ds = mask_to_vis[::ds_factor, ::ds_factor]
            h_ds, w_ds = mask_ds.shape

            print(f"Downsampled mask from {h}x{w} to {h_ds}x{w_ds} for visualization")
            mask_to_vis = mask_ds

    # Convert to numpy for plotting
    mask_np = mask_to_vis.numpy()

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Custom colormap: blues for values (0=light, 1=dark)
    if mask.dtype == torch.bool:
        # For boolean masks: red (masked/False) to blue (attended/True)
        colors = ['#ffcccc', '#99ccff']
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=2)
        vmin, vmax = 0, 1
    else:
        # For float masks: use Blues colormap
        cmap = 'Blues'
        # Determine bounds from data
        vmin = float(mask_np.min())
        vmax = float(mask_np.max())
        # If all values are the same, add a small range
        if vmin == vmax:
            vmin = vmin - 0.1 if vmin > 0 else 0
            vmax = vmax + 0.1 if vmax < 1 else 1

    # Plot the mask
    im = ax.imshow(mask_np, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')

    # Add title with shape information
    mask_title = f"{title}\nShape: {tuple(mask.shape)}"
    if needs_downsampling:
        mask_title += " (Downsampled)"
    ax.set_title(mask_title)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)

    # Add grid lines if appropriate
    h, w = mask_np.shape
    if max(h, w) <= 128:
        # Add major grid lines at section boundaries
        grid_interval = max(1, min(h, w) // 4)
        for i in range(grid_interval, max(h, w), grid_interval):
            if i < h:
                ax.axhline(y=i - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            if i < w:
                ax.axvline(x=i - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved mask visualization to {save_path}")
    plt.close(fig)


def analyze_cp_mask_chunks(cp_outputs, output_dir, layer_idx=0):
    """
    Analyze and visualize each individual CP mask chunk for layer 0.

    Args:
        cp_outputs: Dict containing CP trace outputs
        output_dir: Directory to save visualizations
        layer_idx: Layer to analyze (default: 0)
    """
    if "cp_detailed" not in cp_outputs:
        print(f"\nNo detailed tensor data found. Please run with enhanced tracing.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all detailed data for the specified layer
    layer_data = [
        d for d in cp_outputs.get("cp_detailed", [])
        if d.get("layer_idx", -1) == layer_idx
    ]

    if not layer_data:
        print(f"No detailed data found for layer {layer_idx}")
        return

    rank = cp_outputs.get("cp_detailed", [{}])[0].get("rank", 0)
    print(f"\n===== Analyzing Layer {layer_idx} CP Masks (Rank {rank}) =====")

    # Process each trace entry
    for idx, data in enumerate(layer_data):
        iter_num = data.get("iteration", 0)
        source_rank = data.get("source_rank", (rank - iter_num) % 2)  # Default to calculated source rank

        q = data.get("q")
        k = data.get("k")
        mask = data.get("mask")

        if q is None or k is None:
            continue

        print(f"\nIteration {iter_num} - Source Rank {source_rank}:")

        # Print shapes for q, k, mask
        print(f"  Q shape: {tuple(q.shape) if q is not None else 'None'}")
        print(f"  K shape: {tuple(k.shape) if k is not None else 'None'}")
        print(f"  Mask shape: {tuple(mask.shape) if mask is not None else 'None'}")

        # Visualize the mask
        if mask is not None:
            mask_path = os.path.join(
                output_dir,
                f"layer{layer_idx}_rank{rank}_iter{iter_num}_mask.png"
            )

            # Create a descriptive title
            title = f"Layer {layer_idx}, Rank {rank}, Iter {iter_num}"

            # Visualize raw mask as is
            visualize_raw_mask(mask, mask_path, title)
        else:
            print("  No mask available for this operation")


def analyze_dp_mask(dp_outputs, output_dir, layer_idx=0):
    """
    Analyze and visualize the DP mask for layer 0.

    Args:
        dp_outputs: Dict containing DP trace outputs
        output_dir: Directory to save visualizations
        layer_idx: Layer to analyze (default: 0)
    """
    if "dp_detailed" not in dp_outputs:
        print("\nNo detailed DP tensor data found. Please run with enhanced tracing.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get data for the specified layer
    layer_data = [
        d for d in dp_outputs.get("dp_detailed", [])
        if d.get("layer_idx", -1) == layer_idx
    ]

    if not layer_data:
        print(f"No detailed DP data found for layer {layer_idx}")
        return

    print(f"\n===== Analyzing Layer {layer_idx} DP Mask =====")

    # Take the first entry
    data = layer_data[0]
    q = data.get("q")
    k = data.get("k")
    mask = data.get("mask")

    if q is None or k is None:
        print("Missing query/key data in DP trace")
        return

    # Print shapes
    print(f"  Q shape: {tuple(q.shape) if q is not None else 'None'}")
    print(f"  K shape: {tuple(k.shape) if k is not None else 'None'}")
    print(f"  Mask shape: {tuple(mask.shape) if mask is not None else 'None'}")

    # Visualize the mask
    if mask is not None:
        mask_path = os.path.join(output_dir, f"layer{layer_idx}_dp_mask.png")
        title = f"Layer {layer_idx} DP Mask"
        visualize_raw_mask(mask, mask_path, title)
    else:
        print("  No DP mask available")

def main():
    parser = argparse.ArgumentParser(description="Compare CP and DP attention outputs")
    parser.add_argument("--cp_trace", required=True, help="Path to CP attention trace file")
    parser.add_argument("--dp_trace", required=True, help="Path to DP attention trace file")
    parser.add_argument("--output", default="attention_comparison.txt", help="Output file")
    parser.add_argument("--num_cp_ranks", type=int, default=2, help="Number of CP ranks used")
    parser.add_argument("--load_all_ranks", action="store_true", help="Load traces from all ranks")
    parser.add_argument("--visualize_masks", action="store_true", help="Generate mask visualizations")
    parser.add_argument("--output_dir", default="./mask_visualizations", help="Directory for visualizations")
    args = parser.parse_args()

    import torch  # Import here to avoid requirement when not needed
    import os
    import math

    # Create output directory if needed
    if args.visualize_masks:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load outputs
    cp_outputs = load_attention_outputs(args.cp_trace)
    dp_outputs = load_attention_outputs(args.dp_trace)

    print(f"Loaded CP trace with {len(cp_outputs.get('cp_merged', []))} merged outputs")
    print(f"Loaded DP trace with {len(dp_outputs.get('dp', []))} outputs")

    # Load CP data from all ranks if requested
    cp_outputs_by_rank = {0: cp_outputs}
    if args.load_all_ranks:
        cp_trace_dir = os.path.dirname(args.cp_trace)
        for rank in range(1, args.num_cp_ranks):
            rank_file = os.path.join(cp_trace_dir, f"attention_outputs_rank{rank}.pkl")
            if os.path.exists(rank_file):
                cp_outputs_by_rank[rank] = load_attention_outputs(rank_file)
                print(
                    f"Loaded CP trace for rank {rank} with {len(cp_outputs_by_rank[rank].get('cp_merged', []))} merged outputs")

    # Get all layer indices
    layer_indices = set()
    for out in cp_outputs.get("cp_merged", []):
        layer_indices.add(out.get("layer_idx", -1))
    for out in dp_outputs.get("dp", []):
        layer_indices.add(out.get("layer_idx", -1))
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
    create_chunk_heatmap(results, os.path.join(args.output_dir, "chunk_diff_heatmap.png"))
    print(f"Chunk analysis visualization saved to {os.path.join(args.output_dir, 'chunk_diff_heatmap.png')}")

    # Create overall layer difference visualization
    plt.figure(figsize=(10, 6))
    layer_nums = [r["layer_idx"] for r in results if "diff_percentage" in r]
    diff_pcts = [r["diff_percentage"] for r in results if "diff_percentage" in r]

    if layer_nums and diff_pcts:
        plt.bar(layer_nums, diff_pcts)
        plt.title("Percentage of Different Elements by Layer")
        plt.xlabel("Layer")
        plt.ylabel("% Different Elements")
        plt.savefig(os.path.join(args.output_dir, "layer_diff_pct.png"))
        print(f"Overall visualization saved to {os.path.join(args.output_dir, 'layer_diff_pct.png')}")

    # Run analysis across all loaded ranks
    print("\n=== Analyzing Query/Key/Mask Chunking ===")
    for rank, rank_outputs in cp_outputs_by_rank.items():
        analyze_q_k_mask_chunks(rank_outputs, dp_outputs, args.num_cp_ranks, rank, args.output_dir)

    # Focus on visualizing masks for layer 0 only
    target_layer = 0
    print(f"\n===== Focusing on Layer {target_layer} Mask Analysis =====")

    # Analyze DP mask for layer 0
    analyze_dp_mask(dp_outputs, args.output_dir, target_layer)

    # Analyze CP masks for each rank for layer 0
    for rank, rank_outputs in cp_outputs_by_rank.items():
        analyze_cp_mask_chunks(rank_outputs, args.output_dir, target_layer)

    # Create visualizations of original masks if available
    if args.visualize_masks:
        # Try to get original DP mask
        dp_mask = None
        for detail in dp_outputs.get("dp_detailed", []):
            if "mask" in detail and detail["mask"] is not None:
                dp_mask = detail["mask"]
                break

        if dp_mask is not None:
            # Visualize DP mask directly with the improved raw visualization
            mask_path = os.path.join(args.output_dir, "dp_full_mask.png")
            visualize_raw_mask(dp_mask, mask_path, title="Data Parallel Full Attention Mask")
            print(f"DP mask visualization saved to {mask_path}")

            # We skip the mask reconstruction since we're now directly visualizing the individual masks
            # from each operation in analyze_cp_mask_chunks and analyze_q_k_mask_chunks
        else:
            print("No attention mask found in DP trace for visualization")

if __name__ == "__main__":
    main()