"""
Checkpoint Preloader - Python module to extract tensor metadata from checkpoints
and pass it to C++ extension for preloading into shared memory.
"""
import os
import pickle
import logging
import time
import sys
import signal
import atexit
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import C++ extension for tensor preloading
try:
    import tensor_preloader
    PRELOADER_EXT_AVAILABLE = True
    logger.info("Using tensor_preloader_ext for shared memory loading")
except ImportError:
    PRELOADER_EXT_AVAILABLE = False
    logger.error("tensor_preloader_ext not available! Cannot preload tensors.")


def cleanup_shared_memory():
    """Clean up shared memory segments when the program exits"""
    if PRELOADER_EXT_AVAILABLE:
        try:
            logger.info("Cleaning up shared memory segments...")
            tensor_preloader.cleanup_shared_memory()
            logger.info("Shared memory cleanup complete")
        except Exception as e:
            logger.error(f"Error during shared memory cleanup: {e}")


# Register cleanup function for normal termination
atexit.register(cleanup_shared_memory)


# Signal handler for graceful termination
def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, cleaning up...")
    cleanup_shared_memory()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # termination request


def extract_tensor_metadata(checkpoint_dir: str) -> List[Dict]:
    """
    Extract tensor location metadata from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        List of file entries with tensor information:
        [
            {
                'filepath': str,
                'tensors': [
                    {'offset': int, 'length': int, 'key': str},
                    ...
                ]
            },
            ...
        ]
    """
    metadata_path = os.path.join(checkpoint_dir, ".metadata")

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
                   f"across {len(file_tensors)} files")

        return file_tensors

    except Exception as e:
        logger.error(f"Failed to extract tensor metadata: {e}")
        raise


def preload_checkpoint(checkpoint_dir: str, num_threads: int = 0) -> bool:
    """
    Preload checkpoint tensors into shared memory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        num_threads: Number of threads for parallel loading (0 = auto)

    Returns:
        True if successful, False otherwise
    """
    if not PRELOADER_EXT_AVAILABLE:
        logger.error("Cannot preload tensors: C++ extension not available")
        return False

    try:
        # First clean up any existing shared memory to prevent leaks/conflicts
        cleanup_shared_memory()

        start_time = time.time()

        # Extract tensor metadata
        file_tensors = extract_tensor_metadata(checkpoint_dir)

        # Pass to C++ extension for preloading
        logger.info(f"Starting tensor preloading with C++ extension using {num_threads} threads")
        success = tensor_preloader.preload_tensors(file_tensors, num_threads)

        # Check result
        if success:
            # Get statistics from C++ extension
            stats = tensor_preloader.get_preload_stats()

            elapsed_time = time.time() - start_time
            logger.info(f"Successfully preloaded {stats['preloaded_count']} of {stats['total_count']} "
                       f"tensors ({stats['memory_gb']:.2f} GB) in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.error("Failed to preload tensors")
            return False

    except Exception as e:
        logger.error(f"Error during checkpoint preloading: {e}")
        return False


if __name__ == "__main__":
    # Simple CLI interface
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint_directory> [num_threads]")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cleanup_shared_memory()

    success = preload_checkpoint(checkpoint_dir, num_threads)

    if success:
        print("Preloading successful. Keeping process alive to maintain shared memory...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            cleanup_shared_memory()
    else:
        print("Preloading failed!")
        sys.exit(1)