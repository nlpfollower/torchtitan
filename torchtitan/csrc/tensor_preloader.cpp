#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <chrono>
#include <set>

// Constants matching those in fast_tensor_loader.cpp and model_mapper.cpp
#define SHM_TENSOR_KEY_BASE 10000 // Base key for preloaded tensor segments
#define SHM_TENSOR_META_KEY_BASE 11000 // Base key for tensor metadata

// Mutex for thread-safe console output
std::mutex cout_mutex;

// Global set to track all created shared memory segments for cleanup
std::set<int> created_shm_segments;
std::mutex shm_mutex;

// Structure to store tensor metadata in shared memory
struct TensorMetadata {
    int64_t offset;      // Offset in original file
    int64_t length;      // Length in original file
    int64_t dtype;       // PyTorch dtype
    int64_t ndim;        // Number of dimensions
    int64_t dims[8];     // Dimensions (up to 8D)
    bool is_preloaded;   // Whether tensor was successfully loaded
    char key[256];       // Tensor key (name)
};

// Structure to store file metadata in shared memory
struct FileMetadata {
    int32_t tensor_count;        // Number of tensors in this file
    int32_t preloaded_count;     // Number of successfully preloaded tensors
    size_t tensor_data_size;     // Total size of tensor data
};

// Log levels
enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Current log level
LogLevel current_log_level = INFO;

// Thread-safe logger
void log(LogLevel level, const std::string& message) {
    if (level >= current_log_level) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        const char* level_str = "";
        switch (level) {
            case DEBUG:   level_str = "DEBUG"; break;
            case INFO:    level_str = "INFO"; break;
            case WARNING: level_str = "WARNING"; break;
            case ERROR:   level_str = "ERROR"; break;
        }

        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char time_buf[20];
        std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&now_c));

        std::cout << "[" << time_buf << "][PRELOADER][" << level_str << "] " << message << std::endl;
    }
}

// Generate a unique key for a filepath, using only the basename
int generate_file_key(const std::string& filepath, int base_key) {
    // Extract substring after the last forward slash, if present
    size_t pos = filepath.rfind('/');
    std::string basename = (pos == std::string::npos) ? filepath : filepath.substr(pos + 1);

    // Simple hash function to get a number from the basename
    uint32_t hash = 0;
    for (char c : basename) {
        hash = hash * 31 + c;
    }
    return base_key + (hash % 1000); // Keep keys within a reasonable range
}

// Helper function to register shared memory segment for later cleanup
void register_shm_segment(int shmid) {
    std::lock_guard<std::mutex> lock(shm_mutex);
    created_shm_segments.insert(shmid);
    log(DEBUG, "Registered shared memory segment " + std::to_string(shmid) + " for cleanup");
}

// Read a file in a single chunk
std::vector<char> read_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        throw std::runtime_error("Failed to read file: " + filepath);
    }

    return buffer;
}

// Deserialize a tensor from memory
torch::Tensor load_tensor_from_memory(const char* data, size_t length) {
    try {
        log(DEBUG, "Starting to deserialize tensor of length " + std::to_string(length) + " bytes");

        // Validate input data
        if (data == nullptr) {
            log(ERROR, "Null data pointer passed to load_tensor_from_memory");
            return torch::Tensor();
        }

        if (length == 0) {  // 20GB sanity check
            log(ERROR, "Invalid length passed to load_tensor_from_memory: " + std::to_string(length));
            return torch::Tensor();
        }

        log(DEBUG, "Creating data vector");
        // Create a copy with pre-allocated memory
        std::vector<char> data_vec;
        try {
            data_vec.reserve(length);  // Pre-allocate to avoid reallocation
            log(DEBUG, "Data vector reserved with capacity " + std::to_string(data_vec.capacity()));
            data_vec.assign(data, data + length);
            log(DEBUG, "Data copied to vector, size: " + std::to_string(data_vec.size()));
        } catch (const std::exception& e) {
            log(ERROR, "Exception during data vector creation: " + std::string(e.what()));
            return torch::Tensor();
        }

        log(DEBUG, "Starting pickle_load");
        // This is where most deserialization errors happen
        torch::IValue ivalue;
        try {
            ivalue = torch::pickle_load(data_vec);
            log(DEBUG, "pickle_load completed successfully");
        } catch (const c10::Error& e) {
            log(ERROR, "PyTorch error during pickle_load: " + std::string(e.what()));
            return torch::Tensor();
        } catch (const std::exception& e) {
            log(ERROR, "Standard exception during pickle_load: " + std::string(e.what()));
            return torch::Tensor();
        } catch (...) {
            log(ERROR, "Unknown exception during pickle_load");
            return torch::Tensor();
        }

        // Check if the deserialized object is a tensor
        if (!ivalue.isTensor()) {
            log(ERROR, "Deserialized data is not a tensor");
            return torch::Tensor();
        }

        log(DEBUG, "Converting to tensor");
        torch::Tensor tensor;
        try {
            tensor = ivalue.toTensor().cpu();
            log(DEBUG, "Successfully created tensor with shape: [" +
                 std::to_string(tensor.dim()) + "D, size " +
                 std::to_string(tensor.nbytes()) + " bytes]");
        } catch (const std::exception& e) {
            log(ERROR, "Exception during tensor conversion: " + std::string(e.what()));
            return torch::Tensor();
        }

        return tensor;
    } catch (const c10::Error& e) {
        log(ERROR, "PyTorch error in tensor deserialization: " + std::string(e.what()));
        return torch::Tensor();
    } catch (const std::exception& e) {
        log(ERROR, "Exception in load_tensor_from_memory: " + std::string(e.what()));
        return torch::Tensor();
    } catch (...) {
        log(ERROR, "Unknown exception in load_tensor_from_memory");
        return torch::Tensor();
    }
}

// Global statistics
struct PreloadStats {
    std::atomic<int> total_count{0};
    std::atomic<int> preloaded_count{0};
    std::atomic<size_t> total_memory{0};
} g_stats;

// Helper function to clean up shared memory associated with a file
void cleanup_file_shared_memory(const std::string& filepath) {
    try {
        // Generate keys for this file
        int tensor_meta_key = generate_file_key(filepath, SHM_TENSOR_META_KEY_BASE);
        int tensor_data_key = generate_file_key(filepath, SHM_TENSOR_KEY_BASE);

        // Try to get the segment IDs (to see if they exist)
        int tensor_meta_shmid = shmget(tensor_meta_key, 0, 0666);
        if (tensor_meta_shmid != -1) {
            // Mark for deletion
            if (shmctl(tensor_meta_shmid, IPC_RMID, NULL) == -1) {
                log(WARNING, "Failed to mark tensor metadata shared memory for deletion: " +
                         std::to_string(tensor_meta_key) + ", errno: " + std::to_string(errno));
            } else {
                log(DEBUG, "Marked tensor metadata shared memory for deletion: " +
                        std::to_string(tensor_meta_key));
            }
        }

        int tensor_data_shmid = shmget(tensor_data_key, 0, 0666);
        if (tensor_data_shmid != -1) {
            // Mark for deletion
            if (shmctl(tensor_data_shmid, IPC_RMID, NULL) == -1) {
                log(WARNING, "Failed to mark tensor data shared memory for deletion: " +
                         std::to_string(tensor_data_key) + ", errno: " + std::to_string(errno));
            } else {
                log(DEBUG, "Marked tensor data shared memory for deletion: " +
                        std::to_string(tensor_data_key));
            }
        }
    } catch (const std::exception& e) {
        log(ERROR, "Error cleaning up shared memory for file " + filepath + ": " + e.what());
    }
}

// Preload tensors from a file
// Modify preload_file_tensors to use a fully incremental approach
bool preload_file_tensors(
    const std::string& filepath,
    const std::vector<py::dict>& tensor_infos,
    int num_threads
) {
    try {
        // Count the tensors for this file
        int tensor_count = tensor_infos.size();
        g_stats.total_count += tensor_count;

        if (tensor_count == 0) {
            log(INFO, "No tensors to preload for " + filepath);
            return true;
        }

        log(INFO, "Preloading " + std::to_string(tensor_count) + " tensors from " + filepath);

        // Generate shared memory keys
        int tensor_meta_key = generate_file_key(filepath, SHM_TENSOR_META_KEY_BASE);
        int tensor_data_key = generate_file_key(filepath, SHM_TENSOR_KEY_BASE);

        // Clean up existing segments
        cleanup_file_shared_memory(filepath);

        // Create shared memory for tensor metadata
        int tensor_meta_shmid = shmget(tensor_meta_key,
                                      sizeof(TensorMetadata) * tensor_count + sizeof(FileMetadata),
                                      IPC_CREAT | 0666);
        if (tensor_meta_shmid == -1) {
            log(ERROR, "Failed to create tensor metadata shared memory");
            return false;
        }
        register_shm_segment(tensor_meta_shmid);

        // Attach to tensor metadata segment
        void* shmaddr = shmat(tensor_meta_shmid, NULL, 0);
        if (shmaddr == (void*)-1) {
            log(ERROR, "Failed to attach to tensor metadata shared memory");
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            return false;
        }

        // The layout is: FileMetadata followed by array of TensorMetadata
        FileMetadata* file_metadata = static_cast<FileMetadata*>(shmaddr);
        TensorMetadata* tensor_metadata = reinterpret_cast<TensorMetadata*>(file_metadata + 1);

        // Initialize file metadata
        file_metadata->tensor_count = tensor_count;
        file_metadata->preloaded_count = 0;
        file_metadata->tensor_data_size = 0;

        // First pass: Calculate sizes without loading all tensors at once
        log(INFO, "Starting first pass to calculate sizes");

        // Open the file for streaming reads instead of loading it all at once
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            log(ERROR, "Failed to open file: " + filepath);
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            return false;
        }

        // Initialize metadata from tensor info
        size_t total_tensor_size = 0;
        std::vector<size_t> tensor_sizes(tensor_count, 0);

        // Process in batches of a few tensors at a time
        const int BATCH_SIZE = 8;
        std::vector<char> buffer;

        // First pass - determine sizes by reading small batches
        for (int i = 0; i < tensor_count; i++) {
            try {
                py::dict info = tensor_infos[i];
                int64_t offset = py::cast<int64_t>(info["offset"]);
                int64_t length = py::cast<int64_t>(info["length"]);
                std::string key = py::cast<std::string>(info["key"]);

                log(DEBUG, "First pass: Processing tensor " + std::to_string(i) +
                         " [" + key + "] offset=" + std::to_string(offset) +
                         ", length=" + std::to_string(length));

                // Initialize tensor metadata
                tensor_metadata[i].offset = offset;
                tensor_metadata[i].length = length;
                tensor_metadata[i].is_preloaded = false;
                strncpy(tensor_metadata[i].key, key.c_str(), sizeof(tensor_metadata[i].key) - 1);
                tensor_metadata[i].key[sizeof(tensor_metadata[i].key) - 1] = '\0';

                // We'll manually seek to the correct position for each tensor
                file.seekg(offset);

                // Resize buffer if needed - avoid frequent reallocations
                if (buffer.size() < length) {
                    try {
                        buffer.resize(length);
                    } catch (const std::exception& e) {
                        log(ERROR, "Failed to allocate buffer for tensor " +
                                 std::to_string(i) + ": " + e.what());
                        continue;
                    }
                }

                // Read the tensor data
                if (!file.read(buffer.data(), length)) {
                    log(ERROR, "Failed to read data for tensor " + std::to_string(i));
                    continue;
                }

                // Deserialize the tensor
                torch::Tensor tensor;
                try {
                    tensor = load_tensor_from_memory(buffer.data(), length);
                } catch (const std::exception& e) {
                    log(ERROR, "Failed to deserialize tensor " + std::to_string(i) +
                             ": " + e.what());
                    continue;
                }

                if (tensor.defined()) {
                    if (!tensor.is_contiguous()) {
                        tensor = tensor.contiguous();
                    }

                    // Update metadata
                    tensor_metadata[i].dtype = static_cast<int64_t>(tensor.scalar_type());
                    tensor_metadata[i].ndim = tensor.dim();
                    for (int64_t d = 0; d < tensor.dim() && d < 8; d++) {
                        tensor_metadata[i].dims[d] = tensor.size(d);
                    }

                    // Store size and update total
                    tensor_sizes[i] = tensor.nbytes();
                    total_tensor_size += tensor_sizes[i];

                    log(DEBUG, "Tensor " + std::to_string(i) + " processed, size=" +
                             std::to_string(tensor_sizes[i]) + " bytes");
                } else {
                    log(WARNING, "Failed to deserialize tensor " + std::to_string(i));
                }

                // Free memory explicitly - this forces tensor to be deallocated
                tensor = torch::Tensor();

                // Log progress periodically
                if (i % 10 == 0 || i == tensor_count - 1) {
                    log(INFO, "First pass progress: " + std::to_string(i+1) + "/" +
                             std::to_string(tensor_count) + " tensors");
                }

            } catch (const std::exception& e) {
                log(ERROR, "Exception processing tensor " + std::to_string(i) +
                         " in first pass: " + e.what());
            }
        }

        if (total_tensor_size == 0) {
            log(ERROR, "No valid tensors found in " + filepath);
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            return false;
        }

        // Add some padding to be safe
        total_tensor_size += (total_tensor_size / 20); // Add 5% extra

        log(INFO, "First pass complete. Total tensor data size: " +
                 std::to_string(total_tensor_size / (1024 * 1024)) + "MB");

        // Create shared memory for tensor data
        log(INFO, "Creating shared memory segment of " +
                 std::to_string(total_tensor_size / (1024 * 1024)) + "MB for tensor data");

        int tensor_data_shmid = shmget(tensor_data_key, total_tensor_size, IPC_CREAT | 0666);
        if (tensor_data_shmid == -1) {
            log(ERROR, "Failed to create tensor data shared memory: " +
                     std::to_string(errno));
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            return false;
        }
        register_shm_segment(tensor_data_shmid);

        // Attach to tensor data segment
        void* tensor_data = shmat(tensor_data_shmid, NULL, 0);
        if (tensor_data == (void*)-1) {
            log(ERROR, "Failed to attach to tensor data shared memory");
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            shmctl(tensor_data_shmid, IPC_RMID, NULL);
            return false;
        }

        // Second pass - actually copy tensors to shared memory
        log(INFO, "Starting second pass to copy tensors to shared memory");

        // Reopen file for second pass
        file.close();
        file.open(filepath, std::ios::binary);
        if (!file.is_open()) {
            log(ERROR, "Failed to reopen file for second pass");
            shmdt(shmaddr);
            shmdt(tensor_data);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            shmctl(tensor_data_shmid, IPC_RMID, NULL);
            return false;
        }

        size_t current_offset = 0;
        int success_count = 0;

        for (int i = 0; i < tensor_count; i++) {
            try {
                // Skip if we already know it's invalid
                if (tensor_sizes[i] == 0) {
                    continue;
                }

                int64_t offset = tensor_metadata[i].offset;
                int64_t length = tensor_metadata[i].length;

                log(DEBUG, "Second pass: Processing tensor " + std::to_string(i) +
                         " offset=" + std::to_string(offset) +
                         ", length=" + std::to_string(length));

                // Seek to correct position
                file.seekg(offset);

                // Ensure buffer is large enough
                if (buffer.size() < length) {
                    try {
                        buffer.resize(length);
                    } catch (const std::exception& e) {
                        log(ERROR, "Failed to allocate buffer for tensor " +
                                 std::to_string(i) + ": " + e.what());
                        continue;
                    }
                }

                // Read the tensor data
                if (!file.read(buffer.data(), length)) {
                    log(ERROR, "Failed to read data for tensor " + std::to_string(i));
                    continue;
                }

                // Deserialize and copy directly to shared memory
                torch::Tensor tensor;
                try {
                    tensor = load_tensor_from_memory(buffer.data(), length);
                } catch (const std::exception& e) {
                    log(ERROR, "Failed to deserialize tensor " + std::to_string(i) +
                             ": " + e.what());
                    continue;
                }

                if (tensor.defined()) {
                    if (!tensor.is_contiguous()) {
                        tensor = tensor.contiguous();
                    }

                    // Verify tensor size matches expected size
                    size_t actual_size = tensor.nbytes();
                    if (actual_size != tensor_sizes[i]) {
                        log(WARNING, "Tensor size mismatch for tensor " + std::to_string(i) +
                                   ": expected=" + std::to_string(tensor_sizes[i]) +
                                   ", actual=" + std::to_string(actual_size));

                        // Use the larger of the two sizes
                        tensor_sizes[i] = std::max(tensor_sizes[i], actual_size);
                    }

                    // Check if we have enough space left
                    if (current_offset + tensor_sizes[i] <= total_tensor_size) {
                        // Copy to shared memory
                        memcpy(static_cast<char*>(tensor_data) + current_offset,
                               tensor.data_ptr(),
                               tensor_sizes[i]);

                        // Update metadata
                        tensor_metadata[i].is_preloaded = true;

                        // Move pointer
                        current_offset += tensor_sizes[i];
                        success_count++;
                    } else {
                        log(ERROR, "Not enough shared memory for tensor " + std::to_string(i));
                    }

                    // Free memory explicitly
                    tensor = torch::Tensor();
                }

                // Log progress periodically
                if (i % 10 == 0 || i == tensor_count - 1) {
                    log(INFO, "Second pass progress: " + std::to_string(i+1) + "/" +
                             std::to_string(tensor_count) + " tensors");
                }

            } catch (const std::exception& e) {
                log(ERROR, "Exception processing tensor " + std::to_string(i) +
                         " in second pass: " + e.what());
            }
        }

        // Update final metadata
        file_metadata->preloaded_count = success_count;
        file_metadata->tensor_data_size = current_offset;

        // Update global stats
        g_stats.preloaded_count += success_count;
        g_stats.total_memory += current_offset;

        // Clean up
        file.close();
        shmdt(shmaddr);
        shmdt(tensor_data);

        log(INFO, "Successfully preloaded " + std::to_string(success_count) +
                 " of " + std::to_string(tensor_count) + " tensors from " + filepath +
                 " (" + std::to_string(current_offset / (1024 * 1024)) + "MB)");

        return success_count > 0;

    } catch (const std::exception& e) {
        log(ERROR, "Error preloading tensors from " + filepath + ": " + e.what());
        return false;
    }
}

bool cleanup_shared_memory() {
    log(INFO, "Cleaning up shared memory segments");

    // Get all shared memory segments
    for (int key = SHM_TENSOR_META_KEY_BASE; key < SHM_TENSOR_META_KEY_BASE + 1000; key++) {
        int shmid = shmget(key, 0, 0666);
        if (shmid != -1) {
            shmctl(shmid, IPC_RMID, NULL);
            log(INFO, "Cleaned up metadata segment with key " + std::to_string(key));
        }
    }

    for (int key = SHM_TENSOR_KEY_BASE; key < SHM_TENSOR_KEY_BASE + 1000; key++) {
        int shmid = shmget(key, 0, 0666);
        if (shmid != -1) {
            shmctl(shmid, IPC_RMID, NULL);
            log(INFO, "Cleaned up data segment with key " + std::to_string(key));
        }
    }

    return true;
}

// Main function to preload tensors - extracted from the lambda
bool preload_tensors(py::list file_tensors, int num_threads) {
    try {
        // Reset global statistics
        g_stats.total_count = 0;
        g_stats.preloaded_count = 0;
        g_stats.total_memory = 0;

        log(INFO, "Starting tensor preloading for " +
                 std::to_string(file_tensors.size()) + " files");

        // Process each file
        for (auto& item : file_tensors) {
            py::dict file_entry = py::cast<py::dict>(item);
            std::string filepath = py::cast<std::string>(file_entry["filepath"]);
            py::list tensors = py::cast<py::list>(file_entry["tensors"]);

            // Convert tensors to vector of dicts
            std::vector<py::dict> tensor_infos;
            for (auto& tensor_obj : tensors) {
                tensor_infos.push_back(py::cast<py::dict>(tensor_obj));
            }

            // Preload tensors for this file
            if (!preload_file_tensors(filepath, tensor_infos, num_threads)) {
                log(WARNING, "Failed to preload tensors for " + filepath);
                // Continue with next file
            }
        }

        log(INFO, "Completed tensor preloading: " +
                 std::to_string(g_stats.preloaded_count) + "/" +
                 std::to_string(g_stats.total_count) + " tensors, " +
                 std::to_string(g_stats.total_memory / (1024.0 * 1024.0 * 1024.0)) +
                 "GB shared memory");

        return g_stats.preloaded_count > 0;

    } catch (const std::exception& e) {
        log(ERROR, "Error in preload_tensors: " + std::string(e.what()));
        return false;
    }
}

// Get preloading statistics as a Python dictionary
py::dict get_preload_stats() {
    py::dict stats;
    stats["total_count"] = g_stats.total_count.load();
    stats["preloaded_count"] = g_stats.preloaded_count.load();
    stats["memory_bytes"] = g_stats.total_memory.load();
    stats["memory_gb"] = g_stats.total_memory.load() / (1024.0 * 1024.0 * 1024.0);
    return stats;
}

// Set the log level
void set_log_level(const std::string& level) {
    if (level == "DEBUG") {
        current_log_level = DEBUG;
    } else if (level == "INFO") {
        current_log_level = INFO;
    } else if (level == "WARNING") {
        current_log_level = WARNING;
    } else if (level == "ERROR") {
        current_log_level = ERROR;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Main function to preload tensors
    m.def("preload_tensors", &preload_tensors,
          "Preload tensors from checkpoint files into shared memory",
          py::arg("file_tensors"),
          py::arg("num_threads") = 0);

    // Get preloading statistics
    m.def("get_preload_stats", &get_preload_stats,
          "Get tensor preloading statistics");

    // Set log level
    m.def("set_log_level", &set_log_level,
          "Set the log level",
          py::arg("level"));

    // Clean up shared memory
    m.def("cleanup_shared_memory", &cleanup_shared_memory,
          "Clean up all shared memory segments created by this module");
}