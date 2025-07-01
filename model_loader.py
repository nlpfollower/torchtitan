"""
Checkpoint Preloader - Python module to:
1. Extract tensor metadata from checkpoints
2. Preload tensors into shared memory
"""
import os
import pickle
import logging
import time
import sys
import signal
import atexit
from typing import Dict, List, Any, Optional, Tuple
import torch

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import C++ extensions
try:
    import tensor_preloader
    PRELOADER_EXT_AVAILABLE = True
    logger.info("Using tensor_preloader_ext for shared memory loading")
except ImportError:
    PRELOADER_EXT_AVAILABLE = False
    logger.error("tensor_preloader_ext not available! Cannot preload tensors.")


def cleanup_resources():
    """Clean up resources when the program exits"""
    if PRELOADER_EXT_AVAILABLE:
        try:
            logger.info("Cleaning up shared memory segments...")
            tensor_preloader.cleanup_shared_memory()
            logger.info("Shared memory cleanup complete")
        except Exception as e:
            logger.error(f"Error during shared memory cleanup: {e}")


# Register cleanup function for normal termination
atexit.register(cleanup_resources)


# Signal handler for graceful termination
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_resources()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # termination request


def extract_tensor_metadata(checkpoint_dir: str, metadata_path: Optional[str] = None) -> List[Dict]:
    """
    Extract tensor location metadata from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        metadata_path: Optional path to metadata file. If None, defaults to checkpoint_dir/.metadata

    Returns:
        List of file entries with tensor information
    """
    if metadata_path is None:
        metadata_path = os.path.join(checkpoint_dir, ".metadata")

    # Ensure metadata path is absolute for consistent file operations
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.abspath(metadata_path)

    try:
        # Load the metadata file
        with open(metadata_path, 'rb') as f:
            checkpoint_metadata = pickle.load(f)

        # Extract tensor locations from storage_data
        storage_data = checkpoint_metadata.storage_data

        # Group tensors by file
        file_tensor_map = {}
        for key, storage_info in storage_data.items():
            # Get tensor key (fully qualified name)
            tensor_key = getattr(key, 'fqn', str(key))

            # Get file path and location info
            relative_path = storage_info.relative_path
            full_path = os.path.join(checkpoint_dir, relative_path)
            offset = storage_info.offset
            length = storage_info.length

            # Create entry for this file if needed
            if full_path not in file_tensor_map:
                file_tensor_map[full_path] = []

            # Add tensor info
            file_tensor_map[full_path].append({
                'offset': offset,
                'length': length,
                'key': tensor_key
            })

        # Convert to list format for C++ extension
        file_tensors = [
            {'filepath': filepath, 'tensors': tensors}
            for filepath, tensors in file_tensor_map.items()
        ]

        # Count total tensors
        total_tensors = sum(len(entry['tensors']) for entry in file_tensors)
        logger.info(f"Extracted metadata for {total_tensors} tensors "
                   f"across {len(file_tensors)} files from {metadata_path}")

        return file_tensors

    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract tensor metadata from {metadata_path}: {e}")
        raise


def preload_checkpoint(checkpoint_dir: str, num_threads: int = 0,
                       rank: int = 0, world_size: int = 1,
                       redis_host: str = "localhost", redis_port: int = 6379,
                       run_id: str = None, metadata_path: Optional[str] = None) -> bool:
    """
    Preload checkpoint tensors into shared memory.

    In distributed mode (world_size > 1), the head node (rank 0) reads files
    and broadcasts them to other nodes.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_threads: Number of threads for parallel loading (0 = auto)
        rank: Node rank (0 = root/master)
        world_size: Total number of nodes
        redis_host: Redis server hostname/IP for coordination
        redis_port: Redis server port
        run_id: Unique run ID for coordination (auto-generated if None)
        metadata_path: Optional path to metadata file. If None, defaults to checkpoint_dir/.metadata

    Returns:
        True if successful, False otherwise
    """
    if not PRELOADER_EXT_AVAILABLE:
        logger.error("Cannot preload tensors: C++ extension not available")
        return False

    try:
        import time
        import uuid

        start_time = time.time()

        # Generate a unique run ID if not provided and in distributed mode
        if world_size > 1 and run_id is None:
            logger.error("Run ID is required in distributed mode")
            return False
        elif run_id is None:
            run_id = ""

        PRELOAD_COMPLETE_FILE = "/tmp/tensor_preload_{run_id}_complete"

        # First clean up any existing shared memory to prevent leaks/conflicts
        cleanup_resources()

        # Extract tensor metadata with optional custom metadata path
        file_tensors = extract_tensor_metadata(checkpoint_dir, metadata_path)

        # Pass to C++ extension for preloading with optional distributed parameters
        logger.info(f"Starting tensor preloading (rank {rank}/{world_size}) with C++ extension using {num_threads} threads")
        success = tensor_preloader.preload_tensors(file_tensors, num_threads,
                                                   rank, world_size,
                                                   redis_host, redis_port, run_id)

        # Check result
        if success:
            # Get statistics from C++ extension
            stats = tensor_preloader.get_preload_stats()

            elapsed_time = time.time() - start_time
            logger.info(f"[Rank {rank}] Successfully preloaded {stats['preloaded_count']} of {stats['total_count']} "
                        f"tensors ({stats['memory_gb']:.2f} GB) in {elapsed_time:.2f} seconds")

            with open(PRELOAD_COMPLETE_FILE.format(run_id=run_id), 'w') as f:
                pass
            return True
        else:
            logger.error(f"[Rank {rank}] Failed to preload tensors")
            return False

    except Exception as e:
        logger.error(f"[Rank {rank}] Error during checkpoint preloading: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint Preloader")
    parser.add_argument("checkpoint_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads (0 = auto)")
    parser.add_argument("--rank", type=int, default=0, help="Node rank (0 = root/master)")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis server hostname/IP")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run ID")
    parser.add_argument("--metadata-path", type=str, default=None,
                       help="Path to metadata file (defaults to checkpoint_dir/.metadata)")

    args = parser.parse_args()

    # Clean up existing resources
    cleanup_resources()

    # Attempt to preload the checkpoint
    success = preload_checkpoint(
        args.checkpoint_dir,
        args.threads,
        args.rank,
        args.world_size,
        args.redis_host,
        args.redis_port,
        args.run_id,
        args.metadata_path
    )

    if success:
        print("Preloading successful. Keeping process alive to maintain shared memory...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            cleanup_resources()
    else:
        print("Preloading failed!")
        sys.exit(1)