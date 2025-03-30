import os
import time
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from os import error
from typing import Union, Optional, IO, cast, List, Dict, Any

from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint._extension import ExtensionRegistry
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo, _generate_uuid, _StorageReaderTransforms
from torch.distributed.checkpoint.metadata import MetadataIndex
import torch
from torch import Tensor, Future
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
    from torchtitan.csrc.fast_tensor_loader import FileDataRequest

    FAST_LOADER_AVAILABLE = True
    logger.info("Using fast_tensor_loader extension for improved performance")

    # Set the log level of the C++ extension
    fast_tensor_loader.set_log_level("INFO")
except ImportError:
    FAST_LOADER_AVAILABLE = False
    logger.warning("fast_tensor_loader not available. Will fail if needed")


class OptimizedFileSystemReader(StorageReader):
    """
    Optimized file system reader that uses shared memory for tensor loading.

    This reader accesses files that have been pre-mapped to shared memory
    by the model_mapper process for improved performance.
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

        # Use CPU count if num_threads is not specified
        if num_threads <= 0:
            import multiprocessing
            num_threads = multiprocessing.cpu_count()

        self.num_threads = num_threads

        # Ensure FAST_LOADER_AVAILABLE is True
        if not FAST_LOADER_AVAILABLE:
            raise RuntimeError("fast_tensor_loader is required but not available")

        logger.info(f"Initialized optimized file reader with {self.num_threads} threads")

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """Reset the reader state, optionally changing the checkpoint path."""
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future:
        """Read data according to the load plan."""
        # Start overall timing
        overall_start_time = time.time()

        logger.info(f"Starting to read {len(plan.items)} items")

        # Process byte IO items separately (these are typically small metadata items)
        byte_io_items = [item for item in plan.items if item.type == LoadItemType.BYTE_IO]
        tensor_items = [item for item in plan.items if item.type != LoadItemType.BYTE_IO]

        if len(byte_io_items) > 0:
            raise ValueError("Byte IO items are not supported in this reader")

        # Process tensor items in parallel using shared memory
        tensor_start = time.time()
        self._process_tensor_items(tensor_items, planner)
        tensor_time = time.time() - tensor_start
        logger.info(f"Processed {len(tensor_items)} tensor items in {tensor_time:.6f}s")

        overall_time = time.time() - overall_start_time
        logger.info(f"Overall read_data operation completed in {overall_time:.6f}s")

        fut = Future([torch.device("cuda:0")])
        fut.set_result(None)
        return fut

    def _process_tensor_items(self, items: List[ReadItem], planner: LoadPlanner) -> None:
        """Process tensor items in parallel using shared memory."""
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
        file_requests = []
        for filepath, item_group in file_groups.items():
            for i, (item, item_md) in enumerate(item_group):
                # Create file data request
                request = FileDataRequest()
                request.filepath = str(filepath)
                request.offset = item_md.offset
                request.length = item_md.length
                request.index = len(file_requests)  # Global index in the results array
                request.tensor_offsets = item.storage_offsets
                request.tensor_lengths = item.lengths

                # Store mapping from request index to read item for later
                file_requests.append((request, item))

        # Extract just the request objects for C++ extension
        requests_only = [req for req, _ in file_requests]

        # Process all tensors in parallel using shared memory
        start_time = time.time()

        tensors = fast_tensor_loader.load_tensors_parallel(
            requests_only, self.num_threads
        )

        load_time = time.time() - start_time
        logger.info(f"Loaded {len(tensors)} tensors in {load_time:.6f}s")

        # Process results
        successful = 0
        for i, (_, item) in enumerate(file_requests):
            if i >= len(tensors) or not tensors[i].numel():
                logger.warning(f"Empty or missing tensor for item {i}")
                continue

            try:
                tensor = tensors[i]
                target_tensor = planner.resolve_tensor(item).detach()

                if target_tensor.size() != tensor.size():
                    logger.error(f"Size mismatch: {target_tensor.size()} vs {tensor.size()}")
                    continue

                target_tensor.copy_(tensor)
                planner.commit_tensor(item, target_tensor)
                successful += 1
            except Exception as e:
                logger.error(f"Error processing tensor result {i}: {e}")

        logger.info(f"Successfully processed {successful}/{len(file_requests)} tensors")

    def _create_file_slice(self, file: IO[bytes], offset: int, length: int) -> IO[bytes]:
        """Create a slice of a file with given offset and length."""

        class FileSlice(io.IOBase):
            def __init__(self, base_stream: IO[bytes], start_offset: int, slice_length: int):
                self.base_stream = base_stream
                self.offset = start_offset
                self.length = slice_length
                self.position = 0
                self.base_stream.seek(start_offset)

            def read(self, size=-1):
                if size == -1 or size > self.length - self.position:
                    size = self.length - self.position
                if size <= 0:
                    return b''
                data = self.base_stream.read(size)
                self.position += len(data)
                return data

            def seekable(self):
                return self.base_stream.seekable()

            def seek(self, offset, whence=0):
                if whence == 0:  # absolute
                    new_pos = offset
                elif whence == 1:  # relative
                    new_pos = self.position + offset
                elif whence == 2:  # from end
                    new_pos = self.length - offset
                else:
                    raise ValueError(f"Invalid whence value: {whence}")

                if new_pos < 0:
                    raise ValueError("Cannot seek before start of file slice")
                if new_pos > self.length:
                    new_pos = self.length

                self.position = new_pos
                self.base_stream.seek(self.offset + new_pos)
                return self.position

        return FileSlice(file, offset, length)

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