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

import pydevd_pycharm
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


def extract_cp_submask(cp_mask, cp_q, cp_k, q_matches, k_matches, q_idx, k_idx, chunk_size):
    """
    Extract the correct portion of the CP mask that corresponds to the specified DP region.

    Args:
        cp_mask: The full CP mask tensor
        cp_q: The CP query tensor (used to determine structure)
        cp_k: The CP key tensor (used to determine structure)
        q_matches: List of matches between CP query chunks and DP query chunks
        k_matches: List of matches between CP key chunks and DP key chunks
        q_idx: The DP query chunk index
        k_idx: The DP key chunk index
        chunk_size: Size of each chunk

    Returns:
        The extracted CP submask corresponding to the DP region (q_idx, k_idx)
    """
    # First determine the CP mask dimensions
    if cp_mask.dim() >= 4:
        batch_size, num_heads, q_cp_mask_size, k_cp_mask_size = cp_mask.shape
    else:
        # Handle 2D or 3D masks
        if cp_mask.dim() == 3:
            batch_size, q_cp_mask_size, k_cp_mask_size = cp_mask.shape
        else:
            q_cp_mask_size, k_cp_mask_size = cp_mask.shape
            batch_size = 1

    # Determine CP chunk sizes
    q_n_chunks = cp_q.size(2) // chunk_size
    k_n_chunks = cp_k.size(2) // chunk_size

    # If CP has multiple chunks, we need to find the right mapping
    if q_n_chunks > 1 or k_n_chunks > 1:
        # Map DP chunk indices to CP chunk indices
        q_cp_idx = -1
        k_cp_idx = -1

        # Find which CP query chunk maps to this DP query chunk
        for cp_chunk_idx, dp_chunk_idx, _ in q_matches:
            if dp_chunk_idx == q_idx:
                q_cp_idx = cp_chunk_idx
                break

        # Find which CP key chunk maps to this DP key chunk
        for cp_chunk_idx, dp_chunk_idx, _ in k_matches:
            if dp_chunk_idx == k_idx:
                k_cp_idx = cp_chunk_idx
                break

        # If we found valid mappings
        if q_cp_idx >= 0 and k_cp_idx >= 0:
            # Calculate chunk sizes in CP mask
            q_cp_chunk_size = q_cp_mask_size // q_n_chunks
            k_cp_chunk_size = k_cp_mask_size // k_n_chunks

            # Calculate slice positions
            q_cp_start = q_cp_idx * q_cp_chunk_size
            q_cp_end = (q_cp_idx + 1) * q_cp_chunk_size
            k_cp_start = k_cp_idx * k_cp_chunk_size
            k_cp_end = (k_cp_idx + 1) * k_cp_chunk_size

            # Extract the correct submask
            if cp_mask.dim() >= 4:
                return cp_mask[0, 0, q_cp_start:q_cp_end, k_cp_start:k_cp_end]
            elif cp_mask.dim() == 3:
                return cp_mask[0, q_cp_start:q_cp_end, k_cp_start:k_cp_end]
            else:
                return cp_mask[q_cp_start:q_cp_end, k_cp_start:k_cp_end]

    # If we couldn't find mapping or CP doesn't have multiple chunks,
    # we need to examine the shape to determine how to extract
    if cp_mask.dim() >= 4:
        # If CP mask shape matches DP region shape exactly, use the whole CP mask
        if q_cp_mask_size == chunk_size and k_cp_mask_size == chunk_size:
            return cp_mask[0, 0]
        # If CP mask has dimensions that match multiple chunks, extract the region directly
        elif q_cp_mask_size > chunk_size or k_cp_mask_size > chunk_size:
            # Calculate which portion of the CP mask corresponds to this region
            # For causal attention with CP, the mapping might not be straightforward

            # Try to find a match based on chunk boundaries in CP mask
            if q_cp_mask_size > chunk_size and k_cp_mask_size > chunk_size:
                # This is a multi-chunk CP mask - extract the specific region
                if q_idx < q_n_chunks and k_idx < k_n_chunks:
                    q_start = q_idx * chunk_size // q_n_chunks
                    q_end = (q_idx + 1) * chunk_size // q_n_chunks
                    k_start = k_idx * chunk_size // k_n_chunks
                    k_end = (k_idx + 1) * chunk_size // k_n_chunks
                    return cp_mask[0, 0, q_start:q_end, k_start:k_end]

                # Fallback to using full mask if region doesn't map cleanly
                return cp_mask[0, 0]
            else:
                # One dimension matches chunk size, one doesn't
                return cp_mask[0, 0]

    # Default fallback - use the whole CP mask
    if cp_mask.dim() >= 4:
        return cp_mask[0, 0]
    elif cp_mask.dim() == 3:
        return cp_mask[0]
    else:
        return cp_mask

def analyze_q_k_mask_chunks(cp_outputs, dp_outputs, num_cp_ranks, rank=0, output_dir="./visualizations"):
    """
    Analyze how query, key, and mask chunks are processed in CP vs DP, focusing on layer 0.
    Includes comparison of non-zero positions between CP and DP masks.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import math

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
    dp_output = dp_data.get("output")

    if dp_q is None or dp_k is None:
        print("Missing query/key data in DP trace")
        return

    print(f"\n===== Analyzing Layer {target_layer} CP Chunking (Rank {rank}) =====")

    # Calculate sequence lengths and chunk sizes
    full_seq_len = dp_q.size(2)
    chunk_size = full_seq_len // (2 * num_cp_ranks)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize grid to track chunk processing assignments
    chunk_grid = np.full((2 * num_cp_ranks, 2 * num_cp_ranks), None)

    # Process each CP operation to analyze mask differences
    cp_sums = []
    for detail in sorted(cp_layer, key=lambda x: x.get("iteration", 0)):
        iter_num = detail.get("iteration", 0)
        source_rank = detail.get("source_rank", (rank - iter_num) % num_cp_ranks)
        cp_q = detail.get("q")
        cp_k = detail.get("k")
        cp_mask = detail.get("mask")
        cp_output = detail.get("output")
        cp_sums.append(cp_mask.float().sum().item())

        if cp_q is None or cp_k is None:
            continue

        print(f"\nIteration {iter_num} - Source Rank {source_rank}:")
        print(f"  Q shape: {tuple(cp_q.shape)}, K shape: {tuple(cp_k.shape)}")

        if cp_mask is not None:
            print(f"  Mask shape: {tuple(cp_mask.shape)}")

            # Visualize the raw CP mask
            mask_path = os.path.join(
                output_dir,
                f"cp_mask_layer{target_layer}_rank{rank}_iter{iter_num}.png"
            )
            visualize_raw_mask(cp_mask, mask_path,
                               f"Layer {target_layer}, Rank {rank}, Iter {iter_num} Mask")
            print(f"  CP mask visualization saved to {mask_path}")

        # Empirically determine which chunks are being processed by comparing with DP data
        # We'll compute this by comparing tensors directly
        q_chunks = []
        k_chunks = []

        # Calculate number of chunks in each dimension based on tensor shapes
        q_n_chunks = cp_q.size(2) // chunk_size
        k_n_chunks = cp_k.size(2) // chunk_size

        print(f"  Empirically detecting chunk positions (expecting {q_n_chunks} Q chunks, {k_n_chunks} K chunks)...")

        # Split CP query tensor into chunks and find best match in DP query tensor
        q_matches = []

        # If CP query has multiple chunks (e.g., 8192 with chunk_size 4096)
        if q_n_chunks == 2:
            # Split CP query into chunks
            cp_q_chunks = cp_q.chunk(q_n_chunks, dim=2)

            # For each chunk, find where it matches in the DP query tensor
            for chunk_idx, cp_q_chunk in enumerate(cp_q_chunks):
                # Flatten for easier comparison
                cp_q_flat = cp_q_chunk.flatten()

                best_match_idx = -1
                best_match_diff = float('inf')

                # Compare with each possible chunk in the DP query
                for dp_chunk_idx in range(2 * num_cp_ranks):
                    dp_chunk_start = dp_chunk_idx * chunk_size
                    dp_chunk_end = (dp_chunk_idx + 1) * chunk_size

                    # Extract corresponding chunk from DP tensor
                    dp_q_chunk = dp_q[:, :, dp_chunk_start:dp_chunk_end, :].flatten()

                    # Calculate difference
                    if cp_q_flat.shape == dp_q_chunk.shape:
                        diff = torch.abs(cp_q_flat - dp_q_chunk).mean().item()

                        if diff < best_match_diff:
                            best_match_diff = diff
                            best_match_idx = dp_chunk_idx

                if best_match_idx >= 0:
                    q_matches.append((chunk_idx, best_match_idx, best_match_diff))
                    q_chunks.append(best_match_idx)

            # Log the matches found
            if q_matches:
                q_indices = [m[1] for m in q_matches]
                print(f"    Matched Q chunks {' and '.join(str(i) for i in q_indices)} (diff: {q_matches[0][2]:.6f})")
        else:
            # If CP query has only one chunk (e.g., 4096 with chunk_size 4096)
            # Find which chunk of DP it corresponds to
            cp_q_flat = cp_q.flatten()

            best_match_idx = -1
            best_match_diff = float('inf')

            for dp_chunk_idx in range(2 * num_cp_ranks):
                dp_chunk_start = dp_chunk_idx * chunk_size
                dp_chunk_end = (dp_chunk_idx + 1) * chunk_size

                # Extract corresponding chunk from DP tensor
                dp_q_chunk = dp_q[:, :, dp_chunk_start:dp_chunk_end, :].flatten()

                # Calculate difference
                if cp_q_flat.shape == dp_q_chunk.shape:
                    diff = torch.abs(cp_q_flat - dp_q_chunk).mean().item()

                    if diff < best_match_diff:
                        best_match_diff = diff
                        best_match_idx = dp_chunk_idx

            if best_match_idx >= 0:
                q_chunks.append(best_match_idx)
                print(f"    Matched Q chunk {best_match_idx} (diff: {best_match_diff:.6f})")

        # Now do the same for key chunks
        k_matches = []

        # If CP key has multiple chunks
        if k_n_chunks == 2:
            # Split CP key into chunks
            cp_k_chunks = cp_k.chunk(k_n_chunks, dim=2)

            # For each chunk, find where it matches in the DP key tensor
            for chunk_idx, cp_k_chunk in enumerate(cp_k_chunks):
                # Flatten for easier comparison
                cp_k_flat = cp_k_chunk.flatten()

                best_match_idx = -1
                best_match_diff = float('inf')

                # Compare with each possible chunk in the DP key
                for dp_chunk_idx in range(2 * num_cp_ranks):
                    dp_chunk_start = dp_chunk_idx * chunk_size
                    dp_chunk_end = (dp_chunk_idx + 1) * chunk_size

                    # Extract corresponding chunk from DP tensor
                    dp_k_chunk = dp_k[:, :, dp_chunk_start:dp_chunk_end, :].flatten()

                    # Calculate difference
                    if cp_k_flat.shape == dp_k_chunk.shape:
                        diff = torch.abs(cp_k_flat - dp_k_chunk).mean().item()

                        if diff < best_match_diff:
                            best_match_diff = diff
                            best_match_idx = dp_chunk_idx

                if best_match_idx >= 0:
                    k_matches.append((chunk_idx, best_match_idx, best_match_diff))
                    k_chunks.append(best_match_idx)

            # Log the matches found
            if k_matches:
                k_indices = [m[1] for m in k_matches]
                print(f"    Matched K chunks {' and '.join(str(i) for i in k_indices)} (diff: {k_matches[0][2]:.6f})")
        else:
            # If CP key has only one chunk
            # Find which chunk of DP it corresponds to
            cp_k_flat = cp_k.flatten()

            best_match_idx = -1
            best_match_diff = float('inf')

            for dp_chunk_idx in range(2 * num_cp_ranks):
                dp_chunk_start = dp_chunk_idx * chunk_size
                dp_chunk_end = (dp_chunk_idx + 1) * chunk_size

                # Extract corresponding chunk from DP tensor
                dp_k_chunk = dp_k[:, :, dp_chunk_start:dp_chunk_end, :].flatten()

                # Calculate difference
                if cp_k_flat.shape == dp_k_chunk.shape:
                    diff = torch.abs(cp_k_flat - dp_k_chunk).mean().item()

                    if diff < best_match_diff:
                        best_match_diff = diff
                        best_match_idx = dp_chunk_idx

            if best_match_idx >= 0:
                k_chunks.append(best_match_idx)
                print(f"    Matched K chunk {best_match_idx} (diff: {best_match_diff:.6f})")

        # Process each matched chunk
        print("  Processing chunks:")
        for q_idx in q_chunks:
            q_start = q_idx * chunk_size
            q_end = (q_idx + 1) * chunk_size
            print(f"    Query chunk {q_idx} (positions {q_start}:{q_end})")

            for k_idx in k_chunks:
                k_start = k_idx * chunk_size
                k_end = (k_idx + 1) * chunk_size
                print(f"      � Key chunk {k_idx} (positions {k_start}:{k_end})")

                # Update the processing grid
                chunk_grid[q_idx, k_idx] = (rank, iter_num)

                # Compare with DP mask if available
                if cp_mask is not None and dp_mask is not None:
                    print(f"    Region Q[{q_start}:{q_end}], K[{k_start}:{k_end}]:")

                    # Extract corresponding region from DP mask
                    if dp_mask.dim() >= 4:
                        dp_submask = dp_mask[0, 0, q_start:q_end, k_start:k_end]
                    else:
                        dp_submask = dp_mask[q_start:q_end, k_start:k_end]

                    # Extract corresponding region from CP mask
                    # First determine which part of CP mask corresponds to this region
                    # For CP with multiple chunks, we need to find which chunk this is
                    cp_submask = extract_cp_submask(
                        cp_mask, cp_q, cp_k,
                        q_matches, k_matches,
                        q_idx, k_idx,
                        chunk_size
                    )

                    # Count non-zero positions in both masks
                    dp_values = dp_submask.float().sum().item()
                    cp_values = cp_submask.float().sum().item()

                    # Calculate ratio (avoid division by zero)
                    mask_ratio = cp_values / max(dp_values, 1) if dp_values > 0 else 0

                    print(f"      DP Mask has {dp_values} non-zero positions in this region")
                    print(f"      CP Mask has {cp_values} non-zero positions")
                    print(f"      Ratio CP/DP: {mask_ratio:.2f}")

                    # Create a side-by-side comparison of masks for this region
                    comparison_path = os.path.join(
                        output_dir,
                        f"mask_comparison_layer{target_layer}_rank{rank}_iter{iter_num}_q{q_idx}_k{k_idx}.png"
                    )

                    # Extract q and k ranges for visualization
                    visualize_mask_pair(
                        cp_mask=cp_submask,
                        dp_mask=dp_submask,
                        dp_q_range=(q_start, q_end),
                        dp_k_range=(k_start, k_end),
                        save_path=comparison_path,
                        title=f"Layer {target_layer}, Rank {rank}, Iter {iter_num}: Q{q_idx}, K{k_idx}"
                    )

        # Create a consolidated comparison for this iteration
        if cp_mask is not None and dp_mask is not None:
            print("  Creating consolidated CP vs DP mask comparison:")

            # Create a more comprehensive comparison that highlights all regions being processed
            consolidated_path = os.path.join(
                output_dir,
                f"consolidated_mask_comparison_layer{target_layer}_rank{rank}_iter{iter_num}.png"
            )

            # Compare the full masks with region highlighting
            compare_consolidated_masks_three_panel(
                cp_mask=cp_mask,
                dp_mask=dp_mask,
                q_chunks=q_chunks,
                k_chunks=k_chunks,
                chunk_size=chunk_size,
                save_path=consolidated_path,
                title=f"Layer {target_layer}, Rank {rank}, Iter {iter_num} Mask Comparison"
            )
            print(f"  Consolidated mask comparison saved to {consolidated_path}")

    # Print the chunk processing assignment table
    print_chunk_assignment_table(chunk_grid, num_cp_ranks)
    print(f"Total sum for rank {rank} in the mask is {sum(cp_sums)} and dp mask is {dp_mask.float().sum().item()}")


def compare_consolidated_masks_three_panel(cp_mask, dp_mask, q_chunks, k_chunks, chunk_size, save_path, title):
    """
    Creates a comprehensive three-panel visualization comparing:
    1. CP mask
    2. Concatenated DP mask chunks (middle)
    3. Full DP mask with highlighted regions

    Args:
        cp_mask: Context parallel mask tensor
        dp_mask: Data parallel mask tensor
        q_chunks: List of query chunk indices being processed (empirically determined)
        k_chunks: List of key chunk indices being processed (empirically determined)
        chunk_size: Size of each chunk
        save_path: Where to save the visualization
        title: Title for the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Get original shapes for reference
    print(f"Original shapes - CP: {cp_mask.shape}, DP: {dp_mask.shape}")

    # Handle large masks by downsampling
    max_vis_size = 512  # Maximum size for visualization

    # Determine if downsampling is needed
    cp_max_dim = max(cp_mask.shape[-2:])
    dp_max_dim = max(dp_mask.shape[-2:])
    max_dim = max(cp_max_dim, dp_max_dim)

    ds_factor = max(1, max_dim // max_vis_size)

    # Extract 2D slices from masks
    if cp_mask.dim() >= 4:
        cp_mask_2d = cp_mask[0, 0]
    elif cp_mask.dim() == 3:
        cp_mask_2d = cp_mask[0]
    else:
        cp_mask_2d = cp_mask

    if dp_mask.dim() >= 4:
        dp_mask_2d = dp_mask[0, 0]
    elif dp_mask.dim() == 3:
        dp_mask_2d = dp_mask[0]
    else:
        dp_mask_2d = dp_mask

    # Downsample if needed
    if ds_factor > 1:
        cp_mask_vis = cp_mask_2d[::ds_factor, ::ds_factor]
        dp_mask_vis = dp_mask_2d[::ds_factor, ::ds_factor]
        print(f"Downsampling masks for visualization (factor: {ds_factor})")
        print(f"Final visualization shapes: CP {cp_mask_vis.shape}, DP {dp_mask_vis.shape}")
    else:
        cp_mask_vis = cp_mask_2d
        dp_mask_vis = dp_mask_2d

    # Convert to numpy for visualization
    cp_np = cp_mask_vis.float().cpu().numpy()
    dp_np = dp_mask_vis.float().cpu().numpy()

    # Create figure for visualization - THREE panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # Plot CP mask (first panel)
    im1 = ax1.imshow(cp_np, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title("CP Mask")
    ax1.set_xlabel("Key position")
    ax1.set_ylabel("Query position")
    plt.colorbar(im1, ax=ax1)

    # Create the middle panel - concatenated chunks from DP mask
    # Extract relevant chunks from DP mask and concatenate them
    dp_chunks = []

    for q_idx in q_chunks:
        q_start = q_idx * chunk_size
        q_end = (q_idx + 1) * chunk_size

        for k_idx in k_chunks:
            k_start = k_idx * chunk_size
            k_end = (k_idx + 1) * chunk_size

            # Extract chunk from DP mask
            if dp_mask_2d.shape[0] >= q_end and dp_mask_2d.shape[1] >= k_end:
                chunk = dp_mask_2d[q_start:q_end, k_start:k_end]

                # Downsample if needed
                if ds_factor > 1:
                    chunk = chunk[::ds_factor, ::ds_factor]

                dp_chunks.append((q_idx, k_idx, chunk))

    # Calculate grid dimensions for concatenated view
    num_chunks = len(dp_chunks)
    if num_chunks == 0:
        # If no chunks, create an empty middle panel
        concatenated = np.zeros((1, 1))
    else:
        # Determine a reasonable grid arrangement
        grid_cols = min(len(k_chunks), 2)  # Max 2 columns
        grid_rows = (num_chunks + grid_cols - 1) // grid_cols  # Ceiling division

        # Get single chunk dimensions
        if dp_chunks:
            chunk_height, chunk_width = dp_chunks[0][2].shape
        else:
            chunk_height, chunk_width = 1, 1

        # Create empty grid
        concatenated = np.zeros((grid_rows * chunk_height, grid_cols * chunk_width))

        # Place chunks in grid
        for idx, (q_idx, k_idx, chunk) in enumerate(dp_chunks):
            row = idx // grid_cols
            col = idx % grid_cols

            # Convert chunk to numpy
            chunk_np = chunk.float().cpu().numpy()

            # Place in grid
            row_start = row * chunk_height
            row_end = (row + 1) * chunk_height
            col_start = col * chunk_width
            col_end = (col + 1) * chunk_width

            # Ensure we don't exceed grid bounds
            if row_end <= concatenated.shape[0] and col_end <= concatenated.shape[1]:
                concatenated[row_start:row_end, col_start:col_end] = chunk_np

    # Plot concatenated chunks (middle panel)
    im2 = ax2.imshow(concatenated, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title("Concatenated DP Chunks")
    ax2.set_xlabel("Key position")
    ax2.set_ylabel("Query position")
    plt.colorbar(im2, ax=ax2)

    # Add chunk labels to the concatenated view
    for idx, (q_idx, k_idx, _) in enumerate(dp_chunks):
        row = idx // grid_cols
        col = idx % grid_cols

        # Calculate center position
        center_y = row * chunk_height + chunk_height // 2
        center_x = col * chunk_width + chunk_width // 2

        # Add label
        ax2.text(
            center_x, center_y, f"Q{q_idx},K{k_idx}",
            color='black', ha='center', va='center',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.7)
        )

    # Plot full DP mask with highlights (third panel)
    im3 = ax3.imshow(dp_np, cmap='Blues', vmin=0, vmax=1)
    ax3.set_title(f"DP Mask (Full)\nShape: {tuple(dp_mask.shape)}")
    ax3.set_xlabel("Key position")
    plt.colorbar(im3, ax=ax3)

    # Add rectangles to the DP mask to highlight the regions being processed
    for q_idx in q_chunks:
        q_start = q_idx * chunk_size
        q_end = (q_idx + 1) * chunk_size

        # Scale to visualization coordinates
        q_start_vis = q_start // ds_factor
        q_end_vis = q_end // ds_factor

        for k_idx in k_chunks:
            k_start = k_idx * chunk_size
            k_end = (k_idx + 1) * chunk_size

            # Scale to visualization coordinates
            k_start_vis = k_start // ds_factor
            k_end_vis = k_end // ds_factor

            # Add rectangle to highlight this region
            rect = plt.Rectangle(
                (k_start_vis - 0.5, q_start_vis - 0.5),
                k_end_vis - k_start_vis,
                q_end_vis - q_start_vis,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax3.add_patch(rect)

            # Add label
            ax3.text(
                k_start_vis + (k_end_vis - k_start_vis) // 2,
                q_start_vis + (q_end_vis - q_start_vis) // 2,
                f"Q{q_idx},K{k_idx}",
                color='black',
                ha='center',
                va='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

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


def print_chunk_assignment_table(chunk_grid, num_cp_ranks):
    """
    Print a visualization of chunk assignments as a grid.

    Args:
        chunk_grid: 2D numpy array containing processing information
        num_cp_ranks: Number of CP ranks used
    """
    print("\nChunk Processing Assignment:")
    print("  ", end="")
    for k in range(2 * num_cp_ranks):
        print(f"k{k:<2}", end=" ")
    print()

    for q in range(2 * num_cp_ranks):
        print(f"q{q:<2}", end=" ")
        for k in range(2 * num_cp_ranks):
            cell = chunk_grid[q, k]
            if cell:
                r, i = cell
                print(f"R{r}I{i}", end=" ")
            else:
                print("----", end=" ")
        print()


def visualize_raw_mask(mask, save_path, title="Attention Mask", max_vis_size=2048):
    """
    Visualizes a mask tensor with efficient handling for large masks.

    Args:
        mask: Tensor containing mask values
        save_path: Where to save the plot
        title: Title for the plot
        max_vis_size: Maximum dimension size before downsampling
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

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

            print(f"Downsampled mask from {h}x{w} to {mask_ds.shape[0]}x{mask_ds.shape[1]} for visualization")
            mask_to_vis = mask_ds

    # Convert to numpy for plotting
    mask_np = mask_to_vis.numpy()

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine colormap and limits based on mask type
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_mask_pair(cp_mask, dp_mask, dp_q_range, dp_k_range, save_path, title="Mask Comparison",
                        max_vis_size=2048):
    """
    Visualizes CP and DP masks side by side with their difference, highlighting discrepancies.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    import torch

    # Handle mask being None
    if cp_mask is None or dp_mask is None:
        print(f"Cannot visualize: Mask is None for {title}")
        return

    # Extract the relevant slice from DP mask
    q_start, q_end = dp_q_range
    k_start, k_end = dp_k_range

    # Extract DP submask for the specified region
    if dp_mask.dim() >= 4:
        dp_submask = dp_mask[0, 0, q_start:q_end, k_start:k_end]
        full_dp_mask = dp_mask[0, 0]
    else:
        dp_submask = dp_mask[q_start:q_end, k_start:k_end]
        full_dp_mask = dp_mask

    # Extract CP mask slice - use as-is without transformations
    if cp_mask.dim() >= 4:
        cp_mask_slice = cp_mask[0, 0]
    elif cp_mask.dim() == 3:
        cp_mask_slice = cp_mask[0]
    else:
        cp_mask_slice = cp_mask

    # Convert to float for comparison
    dp_submask = dp_submask.float()
    cp_mask_slice = cp_mask_slice.float()
    full_dp_mask = full_dp_mask.float()

    # Calculate non-zero positions for diagnostics
    cp_nonzero = cp_mask_slice.sum().item()
    dp_nonzero = dp_submask.sum().item()

    # Handle downsampling for visualization
    needs_downsampling = False
    ds_factor = 1

    if max(cp_mask_slice.shape) > max_vis_size or max(dp_submask.shape) > max_vis_size:
        needs_downsampling = True
        max_dim = max(max(cp_mask_slice.shape), max(dp_submask.shape))
        ds_factor = max(1, max_dim // max_vis_size)

        # Downsample for visualization
        cp_vis = cp_mask_slice[::ds_factor, ::ds_factor]
        dp_vis = dp_submask[::ds_factor, ::ds_factor]
    else:
        cp_vis = cp_mask_slice
        dp_vis = dp_submask

    # Convert to numpy for plotting
    cp_np = cp_vis.cpu().numpy()
    dp_np = dp_vis.cpu().numpy()

    # Calculate difference for visualization
    # If shapes don't match, create a highlighted difference view
    has_shape_mismatch = cp_np.shape != dp_np.shape
    if has_shape_mismatch:
        # Create difference visualization showing both masks
        diff_vis = np.zeros(cp_np.shape)
    else:
        # Create direct difference
        diff_vis = cp_np - dp_np

    # Create figure with 4 subplots (CP mask, DP slice, Difference, Discrepancy Highlight)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Set up colormaps
    blues_cmap = 'Blues'
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        "diff_map", ['#ffffff', '#ffcccc', '#ff0000', '#99ccff', '#0000ff'], N=256)

    # Plot CP mask
    im1 = axes[0].imshow(cp_np, cmap=blues_cmap, vmin=0, vmax=1)
    axes[0].set_title(f"CP Mask\nShape: {tuple(cp_mask.shape)}\nNonzero: {cp_nonzero:.0f}")
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")
    plt.colorbar(im1, ax=axes[0])

    # Plot DP mask
    im2 = axes[1].imshow(dp_np, cmap=blues_cmap, vmin=0, vmax=1)
    axes[1].set_title(f"DP Mask\nQ[{q_start}:{q_end}], K[{k_start}:{k_end}]\nNonzero: {dp_nonzero:.0f}")
    axes[1].set_xlabel("Key position")
    plt.colorbar(im2, ax=axes[1])

    # Plot difference
    im3 = axes[2].imshow(diff_vis, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title("Difference (CP - DP)")
    axes[2].set_xlabel("Key position")
    plt.colorbar(im3, ax=axes[2])

    # Create special visualization highlighting discrepancies
    if has_shape_mismatch or abs(cp_nonzero - dp_nonzero) > 0.1:
        # Create custom visualization showing discrepancy regions
        highlight = np.zeros_like(cp_np)
        if cp_nonzero > dp_nonzero:
            highlight_title = "CP has extra non-zero positions"
            highlight_color = 'red'
        else:
            highlight_title = "DP has extra non-zero positions"
            highlight_color = 'blue'

        axes[3].imshow(highlight, cmap='gray', alpha=0.2)
        axes[3].set_title(highlight_title)
        axes[3].set_xlabel("Position")

        # Add text highlighting the discrepancy
        axes[3].text(
            0.5, 0.5,
            f"Discrepancy: {abs(cp_nonzero - dp_nonzero):.0f}\nCP: {cp_nonzero:.0f}\nDP: {dp_nonzero:.0f}",
            ha='center', va='center',
            fontsize=12,
            color=highlight_color,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    else:
        # No discrepancy
        axes[3].imshow(np.zeros_like(cp_np), cmap='gray', alpha=0.2)
        axes[3].set_title("No Discrepancy")
        axes[3].set_xlabel("Position")

    # Add main title
    discrepancy_info = f"Discrepancy: {abs(cp_nonzero - dp_nonzero):.0f}" if abs(
        cp_nonzero - dp_nonzero) > 0.1 else "No discrepancy"
    plt.suptitle(f"{title}\n{discrepancy_info}")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

    # pydevd_pycharm.settrace('localhost', port=6791, stdoutToServer=True, stderrToServer=True)

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
    for out in cp_outputs.get("cp", []):
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