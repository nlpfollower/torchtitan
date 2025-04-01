// fast_tensor_loader.h
#pragma once

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include "tensor_common.h"

namespace py = pybind11;

// Structure to hold file information passed from Python
struct FileDataRequest {
    std::string filepath;
    int64_t offset;
    int64_t length;
    size_t index;
    std::vector<int64_t> tensor_offsets;
    std::vector<int64_t> tensor_lengths;
};

// Structure that includes the destination tensor
struct TensorCopyRequest {
    std::string filepath;
    int64_t offset;
    int64_t length;
    size_t index;
    std::vector<int64_t> tensor_offsets;
    std::vector<int64_t> tensor_lengths;
    torch::Tensor destination_tensor;  // This will hold a reference to the target tensor
};

// Class for accessing data from shared memory
class SharedMemoryAccess {
public:
    SharedMemoryAccess(const std::string& path);
    ~SharedMemoryAccess();

    bool has_preloaded_tensor(size_t offset) const;
    torch::Tensor get_preloaded_tensor(size_t offset) const;
    bool is_valid() const;
    bool has_tensors() const;

private:
    std::string filepath;
    int tensor_meta_shmid;   // For tensor metadata
    int tensor_data_shmid;   // For tensor data
    void* tensor_meta;       // Pointer to tensor metadata
    void* tensor_data;       // Pointer to tensor data
    bool is_attached;
    bool has_preloaded_tensors; // Whether this file has preloaded tensors
};

// Function that loads and copies tensors directly to their destination
bool load_and_copy_tensors_parallel(
    std::vector<TensorCopyRequest>& requests,
    int num_threads = -1
);