import os
import time
import io
import logging
import copy
import concurrent.futures
from typing import Union, Optional, IO, cast, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future as ConcurrentFuture

from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint._extension import ExtensionRegistry
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo, _generate_uuid, _StorageReaderTransforms
from torch.distributed.checkpoint.metadata import MetadataIndex
import torch
from torch import Tensor, Future
import pickle
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, LoadItemType, ReadItem
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed.checkpoint.filesystem import _create_file_view

# Global variable for the extension availability
try:
    import torchtitan.csrc.fast_tensor_loader as fast_tensor_loader

    FAST_LOADER_AVAILABLE = True
    logging.info("Using fast_tensor_loader extension for improved performance")
except ImportError:
    FAST_LOADER_AVAILABLE = False
    logging.warning("fast_tensor_loader not available. Will fail if needed")

# Set up logger
logger = logging.getLogger(__name__)


class MultiLevelParallelFileSystemReader(StorageReader):
    def __init__(
            self,
            path: Union[str, os.PathLike],
            _extension_registry: Optional[ExtensionRegistry] = None,
            num_threads: int = 0,  # 0 means use system default thread count
            batch_size: int = 32,  # Process this many requests in a batch
            max_file_workers: int = 8,  # Maximum number of files to process in parallel
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
        self.batch_size = batch_size
        self.max_file_workers = max_file_workers

        # Ensure FAST_LOADER_AVAILABLE is True
        if not FAST_LOADER_AVAILABLE:
            raise RuntimeError("fast_tensor_loader is required but not available")

        logger.info(f"Initialized multi-level parallel file reader with {self.num_threads} tensor threads "
                    f"and {self.max_file_workers} file workers")

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def _process_batch(self, relative_path: str, batch_reqs: List[ReadItem], planner: LoadPlanner) -> Tuple[int, int]:
        """Process a batch of requests from a single file.

        Returns:
            Tuple[int, int]: (number of successful tensor loads, number of total tensor requests)
        """
        file_path = self.fs.concat_path(self.path, relative_path)
        successful_tensors = 0
        total_tensors = 0

        batch_start_time = time.time()
        try:
            # Open the file once for this batch
            with self.fs.create_stream(file_path, "rb") as stream:
                # Separate BYTE_IO and tensor requests
                byte_io_reqs = [req for req in batch_reqs if req.type == LoadItemType.BYTE_IO]
                tensor_reqs = [req for req in batch_reqs if req.type != LoadItemType.BYTE_IO]

                # Process BYTE_IO requests (not worth parallelizing these small ones)
                if byte_io_reqs:
                    logger.info(f"Processing {len(byte_io_reqs)} BYTE_IO requests sequentially")
                    for req in byte_io_reqs:
                        item_md = self.storage_data[req.storage_index]
                        file_slice = self._slice_file(stream, item_md)
                        transform_from = self.transforms.transform_load_stream(
                            req, item_md.transform_descriptors or (), file_slice
                        )
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)

                # Process tensor requests in parallel
                if tensor_reqs:
                    total_tensors = len(tensor_reqs)
                    logger.info(f"Processing {total_tensors} tensor requests in parallel")

                    # Create a deep copy of the file stream for each request
                    file_views = []
                    offsets_list = []
                    lengths_list = []
                    request_map = []

                    # Prepare data for parallel processing
                    for idx, req in enumerate(tensor_reqs):
                        try:
                            # Get metadata for this request
                            item_md = self.storage_data[req.storage_index]

                            # Create a file slice
                            file_slice = self._slice_file(stream, item_md)

                            # Apply transform
                            transform_from = self.transforms.transform_load_stream(
                                req, item_md.transform_descriptors or (), file_slice
                            )

                            # Add to processing lists
                            file_views.append(transform_from)
                            offsets_list.append(req.storage_offsets)
                            lengths_list.append(req.lengths)
                            request_map.append(idx)
                        except Exception as e:
                            logger.error(f"Error preparing request {idx}: {e}")

                    # Process tensors in parallel if we have any valid requests
                    if file_views:
                        parallel_start = time.time()
                        try:
                            # Adjust threads to not exceed the number of requests
                            local_threads = min(self.num_threads, len(file_views))

                            tensors = fast_tensor_loader.fast_load_tensors_parallel(
                                file_views, offsets_list, lengths_list, local_threads
                            )

                            parallel_time = time.time() - parallel_start
                            logger.info(f"Parallel tensor loading took {parallel_time:.6f} seconds")

                            # Process results
                            for idx, tensor_idx in enumerate(request_map):
                                req = tensor_reqs[tensor_idx]

                                # Skip invalid tensors
                                if idx >= len(tensors) or tensors[idx] is None or not isinstance(tensors[idx],
                                                                                                 torch.Tensor):
                                    logger.warning(f"Invalid tensor for request {tensor_idx}")
                                    continue

                                tensor = tensors[idx]
                                if not tensor.numel():
                                    logger.warning(f"Empty tensor for request {tensor_idx}")
                                    continue

                                try:
                                    # Get the target tensor and copy the loaded data
                                    target_tensor = planner.resolve_tensor(req).detach()
                                    if target_tensor.size() != tensor.size():
                                        logger.error(f"Size mismatch: {target_tensor.size()} vs {tensor.size()}")
                                        continue

                                    target_tensor.copy_(tensor)
                                    planner.commit_tensor(req, target_tensor)
                                    successful_tensors += 1
                                except Exception as e:
                                    logger.error(f"Error processing result for tensor {tensor_idx}: {e}")

                            logger.info(f"Successfully processed {successful_tensors}/{len(file_views)} tensors")
                        except Exception as e:
                            logger.error(f"Error in parallel tensor loading: {e}")
                    else:
                        logger.warning("No valid file views to process")

            logger.info(f"Batch processing completed in {time.time() - batch_start_time:.6f}s")
            return successful_tensors, total_tensors

        except Exception as e:
            logger.error(f"Error processing batch for {relative_path}: {e}")
            return 0, total_tensors

    def _process_file(self, relative_path: str, reqs: List[ReadItem], planner: LoadPlanner) -> Tuple[int, int]:
        """Process all requests for a single file.

        Returns:
            Tuple[int, int]: (number of successful tensor loads, number of total tensor requests)
        """
        file_start_time = time.time()
        logger.info(f"Processing file {relative_path} with {len(reqs)} requests")

        successful_total = 0
        tensor_total = 0

        # Process requests in batches to control memory usage
        for batch_start in range(0, len(reqs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(reqs))
            batch_reqs = reqs[batch_start:batch_end]

            successful, total = self._process_batch(relative_path, batch_reqs, planner)
            successful_total += successful
            tensor_total += total

        file_time = time.time() - file_start_time
        logger.info(f"File {relative_path} processing completed in {file_time:.6f}s")
        return successful_total, tensor_total

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future:
        # Start overall timing
        overall_start_time = time.time()

        # Group requests by file
        grouping_start_time = time.time()
        per_file: Dict[str, List[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)
        grouping_time = time.time() - grouping_start_time
        logger.info(f"Grouping requests by file took {grouping_time:.6f} seconds")

        # Create a thread pool for file-level parallelism
        # Limit the number of concurrent files to avoid overwhelming the system
        num_file_workers = min(self.max_file_workers, len(per_file))
        logger.info(f"Processing {len(per_file)} files with {num_file_workers} concurrent workers")

        total_successful = 0
        total_tensors = 0

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=num_file_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(self._process_file, relative_path, reqs, planner): relative_path
                for relative_path, reqs in per_file.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                relative_path = future_to_file[future]
                try:
                    successful, total = future.result()
                    total_successful += successful
                    total_tensors += total
                except Exception as e:
                    logger.error(f"Exception processing file {relative_path}: {e}")

        overall_time = time.time() - overall_start_time
        logger.info(
            f"Overall read_data operation completed in {overall_time:.6f}s, "
            f"processed {len(plan.items)} requests across {len(per_file)} files, "
            f"successfully loaded {total_successful}/{total_tensors} tensors"
        )

        fut = Future([torch.device("cuda:0")])
        fut.set_result(None)
        return fut

    # Implementing the abstract functions in StorageReader
    def read_metadata(self) -> Metadata:
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
        start_time = time.time()
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None
        elapsed_time = time.time() - start_time
        logger.info(f"Storage reader setup completed in {elapsed_time:.6f} seconds")

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        start_time = time.time()
        result = plan
        elapsed_time = time.time() - start_time
        logger.info(f"Local plan preparation completed in {elapsed_time:.6f} seconds")
        return result

    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        start_time = time.time()
        result = plans
        elapsed_time = time.time() - start_time
        logger.info(f"Global plan preparation completed in {elapsed_time:.6f} seconds")
        return result

    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to load the checkpoint.
        """
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


def create_optimized_reader(
        path: Union[str, os.PathLike],
        extension_registry=None,
        num_tensor_threads: int = 0,  # 0 means use system default
        file_parallelism: int = 0,  # 0 means auto-configure
        batch_size: int = 32
) -> 'MultiLevelParallelFileSystemReader':
    """
    Creates an optimized multi-level parallel file system reader with auto-tuned parameters.

    Args:
        path: Path to the checkpoint directory
        extension_registry: Optional extension registry for transforms
        num_tensor_threads: Number of threads to use for tensor loading (0 = auto)
        file_parallelism: Number of files to process in parallel (0 = auto)
        batch_size: Number of tensors to process in a batch

    Returns:
        An instance of MultiLevelParallelFileSystemReader
    """
    import multiprocessing

    # Auto-configure number of tensor threads if not specified
    if num_tensor_threads <= 0:
        num_tensor_threads = multiprocessing.cpu_count()

    # Auto-configure file parallelism if not specified
    if file_parallelism <= 0:
        # Use CPU count as a starting point, but cap at a reasonable number
        # to avoid overwhelming the I/O system
        cpu_count = multiprocessing.cpu_count()

        # Scale file parallelism based on core count
        if cpu_count <= 4:
            file_parallelism = 2
        elif cpu_count <= 8:
            file_parallelism = 4
        elif cpu_count <= 16:
            file_parallelism = 6
        elif cpu_count <= 32:
            file_parallelism = 8
        else:
            file_parallelism = 10

        # Check if we're in a memory-constrained environment
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            # Adjust file parallelism based on available memory
            if total_memory_gb < 8:
                file_parallelism = min(file_parallelism, 2)
            elif total_memory_gb < 16:
                file_parallelism = min(file_parallelism, 4)
            elif total_memory_gb < 32:
                file_parallelism = min(file_parallelism, 6)
            # For very large memory systems, we can be more aggressive
            elif total_memory_gb > 64:
                file_parallelism = min(file_parallelism + 2, 16)
        except ImportError:
            # If psutil isn't available, stick with CPU-based estimate
            pass

    logging.info(f"Configured parallel reader with {num_tensor_threads} tensor threads, "
                 f"{file_parallelism} file workers, and batch size {batch_size}")

    return MultiLevelParallelFileSystemReader(
        path=path,
        _extension_registry=extension_registry,
        num_threads=num_tensor_threads,
        batch_size=batch_size,
        max_file_workers=file_parallelism
    )