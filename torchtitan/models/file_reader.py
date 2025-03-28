import os
import time
import io
import logging
from typing import Union, Optional, IO, cast

from torch.distributed.checkpoint import StorageReader
from torch.distributed.checkpoint._extension import ExtensionRegistry
from torch.distributed.checkpoint.filesystem import FileSystem, _StorageInfo, _generate_uuid, _StorageReaderTransforms
from torch.distributed.checkpoint.metadata import MetadataIndex
import torch
from torch import Tensor, Future
import pickle
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, LoadItemType, ReadItem
from torch.distributed.checkpoint.metadata import Metadata, StorageMeta
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import _create_file_view

# Set up logger
logger = logging.getLogger(__name__)


class MyFileSystemReader(StorageReader):
    def __init__(
            self,
            path: Union[str, os.PathLike],
            _extension_registry: Optional[ExtensionRegistry] = None,  # EXPERIMENTAL
    ) -> None:
        super().__init__()
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)
        self.storage_data: dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = _generate_uuid()
        self.transforms = _StorageReaderTransforms(_extension_registry)

    def _slice_file(self, file, sinfo: _StorageInfo) -> IO[bytes]:
        return cast(IO[bytes], _create_file_view(file, sinfo.offset, sinfo.length))

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        self.load_id = _generate_uuid()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # Start overall timing
        overall_start_time = time.time()

        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            # Start per-file timing
            file_start_time = time.time()

            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for idx, req in enumerate(reqs):
                    # Start per-request timing
                    req_start_time = time.time()

                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert (
                                target_tensor.size() == tensor.size()
                        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

                    # Log timing for this request
                    req_time = time.time() - req_start_time
                    logger.info(f"Request {idx} in file {relative_path} completed in {req_time:.6f} seconds")

            # Log timing for this file
            file_time = time.time() - file_start_time
            logger.info(f"File {relative_path} processing completed in {file_time:.6f} seconds")

        # Log overall timing
        overall_time = time.time() - overall_start_time
        logger.info(f"Overall read_data operation completed in {overall_time:.6f} seconds")

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # Implementing the abstract function in StorageReader
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

    def prepare_global_plan(self, plans: list[LoadPlan]) -> list[LoadPlan]:
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