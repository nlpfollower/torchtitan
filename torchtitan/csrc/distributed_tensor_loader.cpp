#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>

#include "tensor_common.h"
#include "memory_mapped_file.h"

namespace py = pybind11;

// Class to hold an in-memory file representation
class InMemoryFile {
public:
    InMemoryFile(size_t size) : data(new char[size]), size(size) {}
    ~InMemoryFile() { delete[] data; }

    char* data;
    size_t size;
};

// Cache for in-memory files received through broadcasting
static std::unordered_map<std::string, std::shared_ptr<InMemoryFile>> file_cache;
static std::mutex file_cache_mutex;

// Virtual memory-mapped file implementation for in-memory data
class VirtualMemoryMappedFile : public MemoryMappedFile {
public:
    VirtualMemoryMappedFile(const std::string& path, char* data, size_t size)
        : MemoryMappedFile() {
        filepath = path;
        mapped_data = data;
        file_size = size;
        is_valid = true;
        fd = -1;  // No actual file descriptor

        log(INFO, "Created virtual memory-mapped file for: " + path +
                 " (" + std::to_string(size / (1024 * 1024)) + " MB)");
    }

    // No need to override methods as they'll use our injected members
    ~VirtualMemoryMappedFile() override {
        // Do not unmap or close since we don't own the memory
        mapped_data = nullptr;
        fd = -1;
    }
};

// Initialize the distributed backend (call from Python)
// We let Python handle this completely
bool init_distributed_from_python() {
    log(INFO, "Using Python-initialized distributed backend");
    return true;
}

// Get rank and world size (from Python)
std::tuple<int64_t, int64_t> get_rank_and_world_size_from_python() {
    // This will be replaced with Python's values
    return std::make_tuple(-1, -1);
}

// Broadcast a file from head node to all other nodes (in-memory)
// We'll use PyTorch's native distributed functionality from Python
// This is just the C++ part to handle memory buffers
bool broadcast_file_internal(const std::string& filepath,
                           const torch::Tensor& file_data,
                           int64_t rank,
                           int64_t world_size,
                           int64_t src_rank) {
    try {
        // Skip if only one node
        if (world_size <= 1) {
            log(INFO, "Single node, skipping broadcast for " + filepath);
            return true;
        }

        // Skip if we already have this file cached (non-head nodes)
        if (rank != src_rank) {
            std::lock_guard<std::mutex> lock(file_cache_mutex);
            if (file_cache.find(filepath) != file_cache.end()) {
                log(INFO, "File already cached: " + filepath);
                return true;
            }
        }

        log(INFO, "Processing file: " + filepath +
                 " of size " + std::to_string(file_data.nbytes() / (1024 * 1024)) + "MB");

        if (rank != src_rank) {
            // Non-source rank: store the received tensor data in our file cache
            size_t file_size = file_data.nbytes();

            try {
                // Allocate memory buffer
                auto mem_file = std::make_shared<InMemoryFile>(file_size);

                // Copy data from tensor to our buffer
                std::memcpy(mem_file->data, file_data.data_ptr(), file_size);

                // Store in cache
                std::lock_guard<std::mutex> lock(file_cache_mutex);
                file_cache[filepath] = mem_file;

                log(INFO, "Cached in-memory file: " + filepath +
                         " (" + std::to_string(file_size / (1024 * 1024)) + " MB)");
            } catch (const std::exception& e) {
                log(ERROR, "Error caching file data: " + std::string(e.what()));
                return false;
            }
        }

        return true;
    } catch (const std::exception& e) {
        log(ERROR, "Error in broadcast_file_internal: " + std::string(e.what()));
        return false;
    }
}

// Create a memory-mapped file interface (either real or virtual)
std::shared_ptr<MemoryMappedFile> create_memory_mapped_file(const std::string& filepath, int64_t rank, int64_t world_size) {
    if (world_size <= 1 || rank <= 0) {
        // Head node or single-node setup, use real file
        return std::make_shared<MemoryMappedFile>(filepath);
    }

    // Non-head node: check if we have this file in memory
    std::lock_guard<std::mutex> lock(file_cache_mutex);
    auto it = file_cache.find(filepath);
    if (it != file_cache.end()) {
        // File is in memory, create virtual memory-mapped interface
        return std::make_shared<VirtualMemoryMappedFile>(
            filepath, it->second->data, it->second->size);
    }

    // File is not in memory, fall back to real file (if available)
    log(WARNING, "File not in memory cache, falling back to real file: " + filepath);
    return std::make_shared<MemoryMappedFile>(filepath);
}

// Free memory for in-memory files
void cleanup_memory() {
    log(INFO, "Cleaning up in-memory file cache");
    std::lock_guard<std::mutex> lock(file_cache_mutex);

    size_t total_freed = 0;
    for (const auto& pair : file_cache) {
        total_freed += pair.second->size;
    }

    file_cache.clear();
    log(INFO, "Freed " + std::to_string(total_freed / (1024 * 1024)) + " MB of memory");
}

// Module definition
PYBIND11_MODULE(distributed_tensor_loader, m) {
    m.def("init_distributed_from_python", &init_distributed_from_python,
          "Register that the distributed backend is initialized from Python");

    m.def("broadcast_file_internal", &broadcast_file_internal,
          "Process a file that has been broadcast via PyTorch distributed",
          py::arg("filepath"),
          py::arg("file_data"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("src_rank") = 0);

    m.def("create_memory_mapped_file", &create_memory_mapped_file,
          "Create a memory-mapped file interface (real or virtual)",
          py::arg("filepath"),
          py::arg("rank"),
          py::arg("world_size"));

    m.def("get_rank_and_world_size_from_python", &get_rank_and_world_size_from_python,
          "Stub for Python to replace with actual rank and world size");

    m.def("cleanup_memory", &cleanup_memory,
          "Clean up in-memory file cache");

    // Export VirtualMemoryMappedFile for direct use from Python
    py::class_<VirtualMemoryMappedFile, MemoryMappedFile, std::shared_ptr<VirtualMemoryMappedFile>>(
        m, "VirtualMemoryMappedFile")
        .def(py::init<const std::string&, char*, size_t>());
}