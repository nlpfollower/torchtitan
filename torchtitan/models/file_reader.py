import os
import time
import logging
from typing import Union, Optional, List, Dict

from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint._extension import ExtensionRegistry
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo, _generate_uuid, _StorageReaderTransforms
from torch.distributed.checkpoint.metadata import MetadataIndex
import torch
from torch import Future
import pickle
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, LoadItemType, ReadItem
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Global variable for the extension availability
try:
    import torchtitan.csrc.fast_tensor_loader as fast_tensor_loader
    from torchtitan.csrc.fast_tensor_loader import TensorCopyRequest

    FAST_LOADER_AVAILABLE = True
    logger.info("Using fast_tensor_loader extension with direct GPU copy for optimal performance")

    # Set the log level of the C++ extension
    fast_tensor_loader.set_log_level("INFO")
except ImportError:
    FAST_LOADER_AVAILABLE = False
    logger.warning("fast_tensor_loader not available. Will fail if needed")


class OptimizedFileSystemReader(StorageReader):
    """
    Optimized file system reader that uses shared memory for tensor loading.

    This implementation loads tensors directly to their destination device using
    CUDA streams for optimal performance, eliminating the Python-side copy_ operation.
    """

    def __init__(
            self,
            path: Union[str, os.PathLike],
            _extension_registry: Optional[ExtensionRegistry] = None,
            num_threads: int = 0,  # 0 means use system default thread count
    ) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)

        # Performance tracking variables
        self.total_load_time = 0.0
        self.total_tensor_count = 0

        # Use CPU count if num_threads is not specified
        if num_threads <= 0:
            import multiprocessing
            num_threads = multiprocessing.cpu_count()

        self.num_threads = num_threads

        # Ensure FAST_LOADER_AVAILABLE is True
        if not FAST_LOADER_AVAILABLE:
            raise RuntimeError("fast_tensor_loader with direct GPU copy is required but not available")

        logger.info(f"Initialized optimized file reader with {self.num_threads} threads")

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """Reset the reader state, optionally changing the checkpoint path."""
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

        # Reset performance counters
        self.total_load_time = 0.0
        self.total_tensor_count = 0

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future:
        """Read data according to the load plan, with direct device copy."""
        # Start overall timing
        overall_start_time = time.time()

        logger.info(f"Starting to read {len(plan.items)} items")

        # Process byte IO items separately (these are typically small metadata items)
        byte_io_items = [item for item in plan.items if item.type == LoadItemType.BYTE_IO]
        tensor_items = [item for item in plan.items if item.type != LoadItemType.BYTE_IO]

        if len(byte_io_items) > 0:
            raise ValueError("Byte IO items are not supported in this optimized reader")

        # Process all tensor items in parallel using shared memory with direct device copy
        tensor_start = time.time()
        self._process_tensor_items_direct_copy(tensor_items, planner)
        tensor_time = time.time() - tensor_start

        # Update performance counters
        self.total_load_time += tensor_time
        self.total_tensor_count += len(tensor_items)

        # Calculate and log averages
        avg_time_per_tensor = self.total_load_time / max(1, self.total_tensor_count)
        logger.info(f"Processed {len(tensor_items)} tensor items in {tensor_time:.6f}s "
                    f"(avg: {avg_time_per_tensor * 1000:.2f}ms per tensor)")

        overall_time = time.time() - overall_start_time
        logger.info(f"Overall read_data operation completed in {overall_time:.6f}s")

        fut = Future([torch.device("cuda:0")])
        fut.set_result(None)
        return fut

    def _process_tensor_items_direct_copy(self, items: List[ReadItem], planner: LoadPlanner) -> None:
        """Process tensor items with direct copy to destination device."""
        # Group items by file path to minimize number of file opens
        file_groups = {}
        for item in items:
            item_md = self.storage_data[item.storage_index]
            filepath = self.fs.concat_path(self.path, item_md.relative_path)
            if filepath not in file_groups:
                file_groups[filepath] = []
            file_groups[filepath].append((item, item_md))

        logger.info(f"Grouped tensor items into {len(file_groups)} files")

        # Prepare requests for all files
        all_requests = []
        for filepath, item_group in file_groups.items():
            for item, item_md in item_group:
                try:
                    # Get the destination tensor first - this is the target for the copy
                    destination_tensor = planner.resolve_tensor(item).detach()

                    # Create tensor copy request
                    request = TensorCopyRequest()
                    request.filepath = str(filepath)
                    request.offset = item_md.offset
                    request.length = item_md.length
                    request.index = len(all_requests)
                    request.tensor_offsets = item.storage_offsets
                    request.tensor_lengths = item.lengths
                    request.destination_tensor = destination_tensor  # Pass the actual destination tensor

                    all_requests.append((request, item))
                except Exception as e:
                    logger.error(f"Error preparing tensor request: {e}")

        total_requests = len(all_requests)
        logger.info(f"Processing {total_requests} tensor requests all at once")

        # Extract just the request objects and commit items
        batch_requests = [req for req, _ in all_requests]
        batch_items = [item for _, item in all_requests]

        # Process all tensors in parallel using shared memory with direct copy
        loading_start = time.time()
        success = fast_tensor_loader.load_and_copy_tensors_parallel(
            batch_requests, self.num_threads
        )
        loading_time = time.time() - loading_start

        # Log detailed timing information
        logger.info(f"fast_tensor_loader processing completed in {loading_time:.6f}s "
                   f"({total_requests / loading_time:.1f} tensors/sec)")

        # No need to call copy_ since it's already done in C++
        # Just notify the planner that the tensors are ready
        commit_start = time.time()
        for i, item in enumerate(batch_items):
            try:
                # Get the destination tensor again
                destination_tensor = planner.resolve_tensor(item).detach()

                # Commit the tensor (doesn't actually copy again, just updates planner state)
                planner.commit_tensor(item, destination_tensor)
            except Exception as e:
                logger.error(f"Error committing tensor: {e}")
        commit_time = time.time() - commit_start

        logger.info(f"Planner commit completed in {commit_time:.6f}s")

    # Implementing the abstract functions in StorageReader
    def read_metadata(self) -> Metadata:
        """Read metadata from the checkpoint."""
        start_time = time.time()
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        elapsed_time = time.time() - start_time
        logger.info(f"Metadata read operation completed in {elapsed_time:.6f} seconds")

        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Set up the storage reader with the given metadata."""
        start_time = time.time()
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None
        elapsed_time = time.time() - start_time
        logger.info(f"Storage reader setup completed in {elapsed_time:.6f} seconds")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Prepare the local plan."""
        start_time = time.time()
        result = plan
        elapsed_time = time.time() - start_time
        logger.info(f"Local plan preparation completed in {elapsed_time:.6f} seconds")
        return result

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        """Prepare the global plan."""
        start_time = time.time()
        result = plans
        elapsed_time = time.time() - start_time
        logger.info(f"Global plan preparation completed in {elapsed_time:.6f} seconds")
        return result

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """Return the checkpoint ID that will be used to load the checkpoint."""
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate the checkpoint ID."""
        return FileSystem.validate_checkpoint_id(checkpoint_id)