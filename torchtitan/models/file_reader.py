import os
import time
import io
import logging
import copy
from typing import Union, Optional, IO, cast, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

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


class ParallelFileSystemReader(StorageReader):
    def __init__(
            self,
            path: Union[str, os.PathLike],
            _extension_registry: Optional[ExtensionRegistry] = None,
            num_threads: int = 0,  # 0 means use default (all available cores)
            batch_size: int = 32,  # Process this many requests in a batch
    ) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)
        self.num_threads = num_threads
        self.batch_size = batch_size

        # Ensure FAST_LOADER_AVAILABLE is True
        if not FAST_LOADER_AVAILABLE:
            raise RuntimeError("fast_tensor_loader is required but not available")

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

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

        # Process each file
        for relative_path, reqs in per_file.items():
            file_start_time = time.time()
            filepath = self.fs.concat_path(self.path, relative_path)
            logger.info(f"Processing file {relative_path} with {len(reqs)} requests")

            # Process requests in batches to control memory usage
            for batch_start in range(0, len(reqs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(reqs))
                batch_reqs = reqs[batch_start:batch_end]
                batch_start_time = time.time()

                # Open the file once for this batch
                with self.fs.create_stream(filepath, "rb") as stream:
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
                        logger.info(f"Processing {len(tensor_reqs)} tensor requests in parallel")

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
                                tensors = fast_tensor_loader.fast_load_tensors_parallel(
                                    file_views, offsets_list, lengths_list, self.num_threads
                                )

                                parallel_time = time.time() - parallel_start
                                logger.info(f"Parallel tensor loading took {parallel_time:.6f} seconds")

                                # Process results
                                successful = 0
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
                                        successful += 1
                                    except Exception as e:
                                        logger.error(f"Error processing result for tensor {tensor_idx}: {e}")

                                logger.info(f"Successfully processed {successful}/{len(file_views)} tensors")
                            except Exception as e:
                                logger.error(f"Error in parallel tensor loading: {e}")
                        else:
                            logger.warning("No valid file views to process")

                logger.info(f"Batch processing completed in {time.time() - batch_start_time:.6f}s")

            logger.info(f"File {relative_path} processing completed in {time.time() - file_start_time:.6f}s")

        overall_time = time.time() - overall_start_time
        logger.info(
            f"Overall read_data operation completed in {overall_time:.6f}s, "
            f"processed {len(plan.items)} requests across {len(per_file)} files"
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