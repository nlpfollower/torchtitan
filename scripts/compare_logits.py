#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare logits between two files")
    parser.add_argument("--file1", type=str, required=True,
                        help="Path to first logits file (reference)")
    parser.add_argument("--file2", type=str, required=True,
                        help="Path to second logits file (test)")
    parser.add_argument("--output_dir", type=str, default="./outputs/",
                        help="Directory to save comparison results")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for comparison")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance for comparison")
    parser.add_argument("--plot", action="store_true",
                        help="Generate histograms of differences")
    return parser.parse_args()


def load_logits(filepath):
    """Load logits and metadata from a saved file"""
    try:
        data = torch.load(filepath, map_location=torch.device('cpu'))
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None


def analyze_differences(ref_logits, test_logits, rtol=1e-3, atol=1e-5):
    """Calculate statistics on the differences between two logit tensors"""
    # Flatten tensors for analysis
    ref_flat = ref_logits.flatten()
    test_flat = test_logits.flatten()

    # Calculate absolute and relative differences
    abs_diff = torch.abs(ref_flat - test_flat)
    rel_diff = abs_diff / (torch.abs(ref_flat) + 1e-8)

    # Calculate stats
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    max_rel_diff = torch.max(rel_diff).item()
    mean_rel_diff = torch.mean(rel_diff).item()

    # Check if tensors are close according to tolerances
    is_close = torch.allclose(ref_logits, test_logits, rtol=rtol, atol=atol)

    # Find positions of largest differences
    max_diff_idx = torch.argmax(abs_diff).item()
    orig_shape = ref_logits.shape

    # Get flat index and convert to multi-dimensional index if tensor is multi-dimensional
    flat_indices = np.unravel_index(max_diff_idx, ref_flat.shape)
    if len(orig_shape) > 1:
        # Convert flat index to multi-dimensional index
        multi_indices = np.unravel_index(max_diff_idx, orig_shape)
        pos_str = f"position {multi_indices}"
    else:
        pos_str = f"index {max_diff_idx}"

    # Get values at max difference position
    ref_val = ref_flat[max_diff_idx].item()
    test_val = test_flat[max_diff_idx].item()

    # Calculate number of values exceeding tolerances
    num_exceed_atol = torch.sum(abs_diff > atol).item()
    num_exceed_rtol = torch.sum(rel_diff > rtol).item()

    # Calculate percentage of values exceeding tolerances
    pct_exceed_atol = 100 * num_exceed_atol / ref_flat.numel()
    pct_exceed_rtol = 100 * num_exceed_rtol / ref_flat.numel()

    # Create a numeric mask for values exceeding tolerance
    values_exceed_tolerance = (abs_diff > atol) | (rel_diff > rtol)

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "is_close": is_close,
        "max_diff_pos": pos_str,
        "max_diff_ref_value": ref_val,
        "max_diff_test_value": test_val,
        "num_exceed_atol": num_exceed_atol,
        "num_exceed_rtol": num_exceed_rtol,
        "pct_exceed_atol": pct_exceed_atol,
        "pct_exceed_rtol": pct_exceed_rtol,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "values_exceed_tolerance": values_exceed_tolerance
    }


def plot_difference_histogram(abs_diff, rel_diff, file1_name, file2_name, output_dir):
    """Generate histograms of absolute and relative differences"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare file names for display - extract just the filename without the path
    file1_display = Path(file1_name).name
    file2_display = Path(file2_name).name

    # Plot absolute differences
    abs_diff_np = abs_diff.numpy()
    bins = min(100, max(10, int(np.sqrt(abs_diff.numel()))))
    ax1.hist(abs_diff_np, bins=bins, alpha=0.7)
    ax1.set_title(f'Absolute Differences:\n{file1_display} vs {file2_display}')
    ax1.set_xlabel('Absolute Difference')
    ax1.set_ylabel('Count')
    ax1.set_yscale('log')

    # Plot relative differences
    rel_diff_np = rel_diff.numpy()
    ax2.hist(rel_diff_np, bins=bins, alpha=0.7)
    ax2.set_title(f'Relative Differences:\n{file1_display} vs {file2_display}')
    ax2.set_xlabel('Relative Difference')
    ax2.set_ylabel('Count')
    ax2.set_yscale('log')

    plt.tight_layout()
    output_path = Path(output_dir) / f'diff_histogram.png'
    plt.savefig(output_path)

    # Create a zoomed-in version focused on the important differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Only include differences above a tiny threshold to see the relevant parts
    threshold = 1e-10
    interesting_abs_diffs = abs_diff_np[abs_diff_np > threshold]
    interesting_rel_diffs = rel_diff_np[rel_diff_np > threshold]

    if len(interesting_abs_diffs) > 0:
        ax1.hist(interesting_abs_diffs, bins=bins, alpha=0.7)
        ax1.set_title(f'Significant Absolute Differences (>{threshold}):\n{file1_display} vs {file2_display}')
        ax1.set_xlabel('Absolute Difference')
        ax1.set_ylabel('Count')
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'No significant differences',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)

    if len(interesting_rel_diffs) > 0:
        ax2.hist(interesting_rel_diffs, bins=bins, alpha=0.7)
        ax2.set_title(f'Significant Relative Differences (>{threshold}):\n{file1_display} vs {file2_display}')
        ax2.set_xlabel('Relative Difference')
        ax2.set_ylabel('Count')
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No significant differences',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)

    plt.tight_layout()
    zoom_output_path = Path(output_dir) / f'diff_histogram_zoomed.png'
    plt.savefig(zoom_output_path)
    plt.close()

    return output_path, zoom_output_path


def get_parallelism_config_str(metadata):
    """Create a string describing the parallelism configuration"""
    if 'parallel_config' in metadata:
        config = metadata['parallel_config']
        return f"DP{config['dp_shard']}_CP{config['cp']}_TP{config['tp']}_PP{config['pp']}"
    else:
        return "Unknown"


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both files
    file1_data = load_logits(args.file1)
    file2_data = load_logits(args.file2)

    if file1_data is None or file2_data is None:
        print("Error loading one or both files. Exiting.")
        return

    # Extract logits and metadata
    file1_logits = file1_data["logits"]
    file2_logits = file2_data["logits"]

    file1_metadata = file1_data.get("metadata", {})
    file2_metadata = file2_data.get("metadata", {})

    file1_config = get_parallelism_config_str(file1_metadata)
    file2_config = get_parallelism_config_str(file2_metadata)

    print(f"File 1: {args.file1}")
    print(f"  Config: {file1_config}")
    print(f"  Shape: {file1_logits.shape}")

    print(f"File 2: {args.file2}")
    print(f"  Config: {file2_config}")
    print(f"  Shape: {file2_logits.shape}")

    # Verify tensor shapes match
    if file1_logits.shape != file2_logits.shape:
        print(f"ERROR: Shape mismatch: File 1 {file1_logits.shape} vs File 2 {file2_logits.shape}")
        return

    # Analyze differences
    diff_stats = analyze_differences(file1_logits, file2_logits, args.rtol, args.atol)

    # Plot histograms if requested
    if args.plot:
        output_path, zoom_path = plot_difference_histogram(
            diff_stats["abs_diff"],
            diff_stats["rel_diff"],
            args.file1,
            args.file2,
            output_dir
        )
        print(f"Saved difference histograms to {output_path} and {zoom_path}")

    # Print key results
    status = "PASSED ✅" if diff_stats["is_close"] else "FAILED ❌"
    print("\n=== Comparison Results ===")
    print(f"Overall result: {status}")
    print(f"Max absolute difference: {diff_stats['max_abs_diff']:.6e} at {diff_stats['max_diff_pos']}")
    print(f"  - File 1 value: {diff_stats['max_diff_ref_value']:.6e}")
    print(f"  - File 2 value: {diff_stats['max_diff_test_value']:.6e}")
    print(f"Mean absolute difference: {diff_stats['mean_abs_diff']:.6e}")
    print(f"Max relative difference: {diff_stats['max_rel_diff']:.6e}")
    print(f"Mean relative difference: {diff_stats['mean_rel_diff']:.6e}")
    print(
        f"Values exceeding absolute tolerance ({args.atol:.6e}): {diff_stats['pct_exceed_atol']:.4f}% ({diff_stats['num_exceed_atol']} of {diff_stats['abs_diff'].numel()})")
    print(
        f"Values exceeding relative tolerance ({args.rtol:.6e}): {diff_stats['pct_exceed_rtol']:.4f}% ({diff_stats['num_exceed_rtol']} of {diff_stats['rel_diff'].numel()})")

    # Print timing information if available
    if 'elapsed_time' in file1_metadata and 'elapsed_time' in file2_metadata:
        time1 = file1_metadata['elapsed_time']
        time2 = file2_metadata['elapsed_time']
        speedup = time1 / time2 if time2 > 0 else 0

        print("\n=== Performance Comparison ===")
        print(f"File 1 execution time: {time1:.3f}s")
        print(f"File 2 execution time: {time2:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

    # Save detailed results to a text file
    results_file = output_dir / "comparison_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Comparison between:\n")
        f.write(f"  File 1: {args.file1}\n")
        f.write(f"  File 2: {args.file2}\n\n")

        f.write(f"=== Comparison Results ===\n")
        f.write(f"Overall result: {status}\n")
        f.write(f"Max absolute difference: {diff_stats['max_abs_diff']:.6e} at {diff_stats['max_diff_pos']}\n")
        f.write(f"  - File 1 value: {diff_stats['max_diff_ref_value']:.6e}\n")
        f.write(f"  - File 2 value: {diff_stats['max_diff_test_value']:.6e}\n")
        f.write(f"Mean absolute difference: {diff_stats['mean_abs_diff']:.6e}\n")
        f.write(f"Max relative difference: {diff_stats['max_rel_diff']:.6e}\n")
        f.write(f"Mean relative difference: {diff_stats['mean_rel_diff']:.6e}\n")
        f.write(
            f"Values exceeding absolute tolerance ({args.atol:.6e}): {diff_stats['pct_exceed_atol']:.4f}% ({diff_stats['num_exceed_atol']} of {diff_stats['abs_diff'].numel()})\n")
        f.write(
            f"Values exceeding relative tolerance ({args.rtol:.6e}): {diff_stats['pct_exceed_rtol']:.4f}% ({diff_stats['num_exceed_rtol']} of {diff_stats['rel_diff'].numel()})\n")

        if 'elapsed_time' in file1_metadata and 'elapsed_time' in file2_metadata:
            f.write(f"\n=== Performance Comparison ===\n")
            f.write(f"File 1 execution time: {time1:.3f}s\n")
            f.write(f"File 2 execution time: {time2:.3f}s\n")
            f.write(f"Speedup: {speedup:.2f}x\n")

    print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    main()