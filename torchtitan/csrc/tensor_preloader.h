#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>
#include "tensor_common.h"
#include "thread_pool.h"
#include "memory_mapped_file.h"
#include "shared_memory.h"

namespace py = pybind11;

// Struct for tensor sizing task
struct TensorSizingTask {
    int index;
    int64_t offset;
    int64_t length;
    std::string key;
};

// Struct for tensor loading task
struct TensorLoadingTask {
    int index;
    int64_t offset;
    int64_t length;
    std::string key;
    size_t size;
    size_t shm_offset;
};

// Function to analyze tensor and get its size
size_t analyze_tensor_size(const char* tensor_data, size_t length, 
                          TensorMetadata& metadata);

// Function to load tensor into shared memory
bool load_tensor_to_shared_memory(const char* tensor_data, size_t length,
                                void* shm_data, size_t shm_offset);

// Primary tensor loading function
bool preload_file_tensors(const std::string& filepath,
                        const std::vector<py::dict>& tensor_infos,
                        int num_threads,
                        GlooFileBroadcast* broadcaster);

// Global preloader function
bool preload_tensors(py::list file_tensors, int num_threads,
                    int rank, int world_size,
                    const std::string& redis_host,
                    int redis_port,
                    const std::string& run_id);

// Get preloading statistics
py::dict get_preload_stats();