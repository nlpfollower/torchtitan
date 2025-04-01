#pragma once

#include <torch/extension.h>
#include <iostream>
#include <chrono>
#include <mutex>
#include <string>
#include <atomic>
#include <sys/shm.h>

// Constants for shared memory keys
#define SHM_TENSOR_KEY_BASE 10000
#define SHM_TENSOR_META_KEY_BASE 11000

// Logging utilities
enum LogLevel { DEBUG, INFO, WARNING, ERROR };

// Structure definitions
struct TensorMetadata {
    int64_t offset;      // Offset in original file
    int64_t length;      // Length in original file
    int64_t dtype;       // PyTorch dtype
    int64_t ndim;        // Number of dimensions
    int64_t dims[8];     // Dimensions (up to 8D)
    bool is_preloaded;   // Whether tensor was successfully loaded
    char key[256];       // Tensor key (name)
    size_t shm_offset;   // Offset within shared memory segment
};

struct FileMetadata {
    int32_t tensor_count;        // Number of tensors in this file
    int32_t preloaded_count;     // Number of successfully preloaded tensors
    size_t tensor_data_size;     // Total size of tensor data
};

// Thread-safe logger declaration
void log(LogLevel level, const std::string& message);
void set_log_level(const std::string& level);

// Utility functions
int generate_file_key(const std::string& filepath, int base_key);