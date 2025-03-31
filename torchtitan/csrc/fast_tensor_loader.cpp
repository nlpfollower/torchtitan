// fast_tensor_loader.cpp
// Optimized tensor loader that uses shared memory for fast access

#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

// Global constants
#define SHM_META_KEY_BASE 9000  // Base key for metadata segments
#define SHM_DATA_KEY_BASE 10000 // Base key for data segments

// Mutex for thread-safe console output
std::mutex cout_mutex;

// Log levels
enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

// Current log level (can be changed at runtime)
LogLevel current_log_level = INFO;

// Function for logging with thread safety and log levels
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

        std::cout << "[" << time_buf << "][" << level_str << "] " << message << std::endl;
    }
}

// Structure to store file metadata in shared memory
struct FileMetadata {
    size_t file_size;
    uint32_t checksum;
    char filepath[256];  // Fixed-size buffer for filepath
};

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

// Class for accessing data from shared memory
class SharedMemoryAccess {
private:
    std::string filepath;
    int meta_shmid;
    int data_shmid;
    FileMetadata* metadata;
    void* data;
    bool is_attached;

public:
    SharedMemoryAccess(const std::string& path)
        : filepath(path), meta_shmid(-1), data_shmid(-1),
          metadata(nullptr), data(nullptr), is_attached(false)
    {
        // Generate keys based on filepath
        int meta_key = generate_file_key(filepath, SHM_META_KEY_BASE);
        int data_key = generate_file_key(filepath, SHM_DATA_KEY_BASE);

        // Access metadata shared memory
        meta_shmid = shmget(meta_key, sizeof(FileMetadata), 0666);
        if (meta_shmid == -1) {
            throw std::runtime_error("Failed to find metadata shared memory for " + filepath +
                                     " (key=" + std::to_string(meta_key) +
                                     "). Is model_mapper running?");
        }

        // Attach to metadata
        metadata = (FileMetadata*)shmat(meta_shmid, NULL, 0);
        if (metadata == (void*)-1) {
            throw std::runtime_error("Failed to attach to metadata shared memory for " + filepath);
        }

        // Check if the filepath in metadata matches our request
        if (std::string(metadata->filepath) != filepath) {
            shmdt(metadata);
            throw std::runtime_error("Filepath mismatch in shared memory metadata. Expected: " +
                                     filepath + ", Got: " + metadata->filepath);
        }

        // Access data shared memory
        data_shmid = shmget(data_key, metadata->file_size, 0666);
        if (data_shmid == -1) {
            shmdt(metadata);
            throw std::runtime_error("Failed to find data shared memory for " + filepath +
                                     " (key=" + std::to_string(data_key) + ")");
        }

        // Attach to data
        data = shmat(data_shmid, NULL, 0);
        if (data == (void*)-1) {
            shmdt(metadata);
            throw std::runtime_error("Failed to attach to data shared memory for " + filepath);
        }

        is_attached = true;
        log(DEBUG, "Successfully attached to shared memory for " + filepath);
    }

    ~SharedMemoryAccess() {
        if (is_attached) {
            if (data != nullptr) {
                shmdt(data);
            }
            if (metadata != nullptr) {
                shmdt(metadata);
            }
        }
    }

    // Get a slice of data from the shared memory
    std::string get_data_slice(size_t offset, size_t length) const {
        if (!is_attached) {
            throw std::runtime_error("Not attached to shared memory");
        }

        // Log the access attempt
        log(DEBUG, "Accessing shared memory with offset=" + std::to_string(offset) +
                  " length=" + std::to_string(length) +
                  " for file=" + filepath);

        // Verify bounds
        if (offset + length > metadata->file_size) {
            throw std::runtime_error("Requested region exceeds file size");
        }

        // Create a string from the data slice
        char* data_ptr = static_cast<char*>(data) + offset;
        return std::string(data_ptr, length);
    }

    // Get file size from metadata
    size_t get_file_size() const {
        if (!is_attached || metadata == nullptr) {
            throw std::runtime_error("Not attached to shared memory");
        }
        return metadata->file_size;
    }

    // Check if we're successfully attached
    bool is_valid() const {
        return is_attached && metadata != nullptr && data != nullptr;
    }
};

// Function for loading tensor from memory buffer
torch::Tensor load_tensor_from_memory(
    const std::string& data,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& lengths
) {
    try {
        // Convert string data to vector<char> as required by pickle_load
        std::vector<char> data_vec(data.begin(), data.end());

        // Use pickle_load with the correct data type
        torch::IValue ivalue = torch::pickle_load(data_vec);

        // Try to convert to tensor
        if (!ivalue.isTensor()) {
            log(ERROR, "Loaded data is not a tensor");
            return torch::Tensor();
        }

        torch::Tensor tensor = ivalue.toTensor();

        // Apply narrowing if needed
        if (!offsets.empty() && !lengths.empty()) {
            for (size_t i = 0; i < offsets.size() && i < static_cast<size_t>(tensor.dim()); i++) {
                tensor = tensor.narrow(static_cast<int64_t>(i), offsets[i], lengths[i]);
            }
        }

        return tensor;
    } catch (const c10::Error& e) {
        log(ERROR, "Error in tensor deserialization: " + std::string(e.what()));
        return torch::Tensor(); // Return empty tensor
    } catch (const std::exception& e) {
        log(ERROR, "Exception in load_tensor_from_memory: " + std::string(e.what()));
        return torch::Tensor(); // Return empty tensor
    }
}

// Structure to hold file information passed from Python
struct FileDataRequest {
    std::string filepath;
    int64_t offset;
    int64_t length;
    size_t index;
    std::vector<int64_t> tensor_offsets;
    std::vector<int64_t> tensor_lengths;
};

// Parallel processing function using shared memory
std::vector<torch::Tensor> load_tensors_parallel(
    const std::vector<FileDataRequest>& file_requests,
    int num_threads = -1
) {
    // Determine thread count
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    num_threads = std::max(1, num_threads);

    size_t request_count = file_requests.size();
    log(INFO, "Processing " + std::to_string(request_count) + " tensors with " +
         std::to_string(num_threads) + " threads (PID: " +
         std::to_string(getpid()) + ")");

    // Prepare output vector
    std::vector<torch::Tensor> results(request_count);

    // Process requests in parallel
    std::atomic<size_t> request_index(0);
    std::vector<std::thread> threads;

    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&file_requests, &results, &request_index]() {
            // Get thread ID for logging
            auto thread_id = std::this_thread::get_id();
            std::stringstream ss;
            ss << thread_id;
            std::string thread_id_str = ss.str();

            while (true) {
                // Get next request atomically
                size_t idx = request_index.fetch_add(1);
                if (idx >= file_requests.size()) {
                    break;  // No more requests
                }

                const FileDataRequest& req = file_requests[idx];
                auto start_time = std::chrono::high_resolution_clock::now();

                try {
                    // Access the file data from shared memory
                    SharedMemoryAccess shm_access(req.filepath);

                    // Get the slice of data we need
                    std::string data = shm_access.get_data_slice(req.offset, req.length);

                    if (!data.empty()) {
                        // Process the data
                        torch::Tensor tensor = load_tensor_from_memory(
                            data, req.tensor_offsets, req.tensor_lengths
                        );

                        // Store the result
                        results[req.index] = tensor;

                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time
                        ).count();

                        log(DEBUG, "Thread " + thread_id_str + " processed tensor " +
                                  std::to_string(req.index) + " in " + std::to_string(duration) + "ms");
                    } else {
                        log(WARNING, "Thread " + thread_id_str + " got empty data for file " +
                                    req.filepath + " at index " + std::to_string(req.index));
                    }
                } catch (const std::exception& e) {
                    log(ERROR, "Thread " + thread_id_str + " error processing file " +
                                req.filepath + " at index " + std::to_string(req.index) +
                                ": " + e.what());
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Check for any unprocessed tensors
    size_t failed_count = 0;
    for (size_t i = 0; i < request_count; ++i) {
        if (!results[i].defined()) {
            failed_count++;
        }
    }

    if (failed_count > 0) {
        log(WARNING, std::to_string(failed_count) + " tensors failed to process");
    } else {
        log(INFO, "All tensors processed successfully");
    }

    return results;
}


// New structure that includes the destination tensor
struct TensorCopyRequest {
    std::string filepath;
    int64_t offset;
    int64_t length;
    size_t index;
    std::vector<int64_t> tensor_offsets;
    std::vector<int64_t> tensor_lengths;
    torch::Tensor destination_tensor;  // This will hold a reference to the target tensor
};

// Function that loads and copies tensors directly to their destination
bool load_and_copy_tensors_parallel(
    std::vector<TensorCopyRequest>& requests,
    int num_threads = -1
) {
    // Determine thread count
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    num_threads = std::max(1, num_threads);

    size_t request_count = requests.size();
    log(INFO, "Processing " + std::to_string(request_count) + " tensors with " +
         std::to_string(num_threads) + " threads (PID: " +
         std::to_string(getpid()) + ")");

    // Process requests in parallel
    std::atomic<size_t> request_index(0);
    std::atomic<size_t> success_count(0);
    std::vector<std::thread> threads;

    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&requests, &request_index, &success_count]() {
            // Get thread ID for logging
            auto thread_id = std::this_thread::get_id();
            std::stringstream ss;
            ss << thread_id;
            std::string thread_id_str = ss.str();

            while (true) {
                // Get next request atomically
                size_t idx = request_index.fetch_add(1);
                if (idx >= requests.size()) {
                    break;  // No more requests
                }

                TensorCopyRequest& req = requests[idx];
                auto start_time = std::chrono::high_resolution_clock::now();

                try {
                    // Access the file data from shared memory
                    SharedMemoryAccess shm_access(req.filepath);

                    // Get the slice of data we need
                    std::string data = shm_access.get_data_slice(req.offset, req.length);

                    if (!data.empty()) {
                        // Process the data - load into CPU tensor
                        torch::Tensor cpu_tensor = load_tensor_from_memory(
                            data, req.tensor_offsets, req.tensor_lengths
                        );

                        if (cpu_tensor.defined() && cpu_tensor.numel() > 0) {
                            // Verify tensor sizes match
                            if (cpu_tensor.sizes() != req.destination_tensor.sizes()) {
                                log(ERROR, "Size mismatch for tensor " + std::to_string(idx) +
                                          ": source " + std::to_string(cpu_tensor.numel()) +
                                          " vs destination " + std::to_string(req.destination_tensor.numel()));
                                continue;
                            }

                            // Simpler approach: just use copy_ without explicit stream handling
                            // For CUDA tensors, this will use the current CUDA stream
                            req.destination_tensor.copy_(cpu_tensor);
                            success_count++;

                            auto end_time = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                end_time - start_time
                            ).count();

                            log(DEBUG, "Thread " + thread_id_str + " processed tensor " +
                                      std::to_string(idx) + " in " + std::to_string(duration) + "ms");
                        } else {
                            log(WARNING, "Failed to load tensor data for file " + req.filepath);
                        }
                    } else {
                        log(WARNING, "Thread " + thread_id_str + " got empty data for file " +
                                    req.filepath + " at index " + std::to_string(idx));
                    }
                } catch (const std::exception& e) {
                    log(ERROR, "Thread " + thread_id_str + " error processing file " +
                                req.filepath + " at index " + std::to_string(idx) +
                                ": " + e.what());
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Report completion
    log(INFO, "Completed processing " + std::to_string(success_count.load()) +
             " out of " + std::to_string(request_count) + " tensors");

    return success_count.load() == request_count;
}

// Set the log level (can be called from Python)
void set_log_level(const std::string& level) {
    if (level == "DEBUG") {
        current_log_level = DEBUG;
    } else if (level == "INFO") {
        current_log_level = INFO;
    } else if (level == "WARNING") {
        current_log_level = WARNING;
    } else if (level == "ERROR") {
        current_log_level = ERROR;
    } else {
        log(WARNING, "Unknown log level: " + level + ". Using INFO.");
        current_log_level = INFO;
    }
    log(INFO, "Log level set to " + level);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Define the FileDataRequest class for Python
    py::class_<FileDataRequest>(m, "FileDataRequest")
        .def(py::init<>())
        .def_readwrite("filepath", &FileDataRequest::filepath)
        .def_readwrite("offset", &FileDataRequest::offset)
        .def_readwrite("length", &FileDataRequest::length)
        .def_readwrite("index", &FileDataRequest::index)
        .def_readwrite("tensor_offsets", &FileDataRequest::tensor_offsets)
        .def_readwrite("tensor_lengths", &FileDataRequest::tensor_lengths);

    // Expose the load_tensor_from_memory function
    m.def("load_tensor_from_memory", &load_tensor_from_memory,
          "Load tensor from memory buffer",
          py::arg("data"),
          py::arg("offsets") = std::vector<int64_t>(),
          py::arg("lengths") = std::vector<int64_t>());

    // Expose the load_tensors_parallel function
    m.def("load_tensors_parallel", &load_tensors_parallel,
          "Load multiple tensors in parallel using shared memory",
          py::arg("file_requests"),
          py::arg("num_threads") = -1);

    // Expose the set_log_level function
    m.def("set_log_level", &set_log_level,
          "Set the log level (DEBUG, INFO, WARNING, ERROR)",
          py::arg("level"));

    // Add to PYBIND11_MODULE
    py::class_<TensorCopyRequest>(m, "TensorCopyRequest")
        .def(py::init<>())
        .def_readwrite("filepath", &TensorCopyRequest::filepath)
        .def_readwrite("offset", &TensorCopyRequest::offset)
        .def_readwrite("length", &TensorCopyRequest::length)
        .def_readwrite("index", &TensorCopyRequest::index)
        .def_readwrite("tensor_offsets", &TensorCopyRequest::tensor_offsets)
        .def_readwrite("tensor_lengths", &TensorCopyRequest::tensor_lengths)
        .def_readwrite("destination_tensor", &TensorCopyRequest::destination_tensor);

    m.def("load_and_copy_tensors_parallel", &load_and_copy_tensors_parallel,
        "Load and directly copy multiple tensors to their destinations in parallel",
        py::arg("requests"),
        py::arg("num_threads") = -1);
}