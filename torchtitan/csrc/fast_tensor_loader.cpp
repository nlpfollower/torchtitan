#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
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

// Global constants - match tensor_preloader.cpp
#define SHM_TENSOR_KEY_BASE 10000 // Base key for preloaded tensor segments
#define SHM_TENSOR_META_KEY_BASE 11000 // Base key for tensor metadata

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

// Function to set the log level
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
        throw std::runtime_error("Invalid log level: " + level +
                                 ". Valid levels are DEBUG, INFO, WARNING, ERROR");
    }
}

// Structure to store tensor metadata in shared memory
struct TensorMetadata {
    int64_t offset;          // Offset in original file
    int64_t length;          // Length in original file
    int64_t dtype;           // PyTorch dtype
    int64_t ndim;            // Number of dimensions
    int64_t dims[8];         // Dimensions (up to 8D)
    bool is_preloaded;       // Whether tensor was successfully loaded
    char key[256];           // Tensor key (name)
    size_t shm_offset;       // NEW: Offset within shared memory segment
};

// Structure for file metadata with tensor info
struct FileMetadataWithTensors {
    int32_t tensor_count;        // Number of tensors in this file
    int32_t preloaded_count;     // Number of successfully preloaded tensors
    size_t tensor_data_size;     // Total size of tensor data
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
    int tensor_meta_shmid;   // For tensor metadata
    int tensor_data_shmid;   // For tensor data
    void* tensor_meta;       // Pointer to tensor metadata
    void* tensor_data;       // Pointer to tensor data
    bool is_attached;
    bool has_preloaded_tensors; // Whether this file has preloaded tensors

public:
    SharedMemoryAccess(const std::string& path)
        : filepath(path),
          tensor_meta_shmid(-1), tensor_data_shmid(-1),
          tensor_meta(nullptr), tensor_data(nullptr),
          is_attached(false), has_preloaded_tensors(false)
    {
        // Generate keys based on filepath
        int tensor_meta_key = generate_file_key(filepath, SHM_TENSOR_META_KEY_BASE);
        int tensor_data_key = generate_file_key(filepath, SHM_TENSOR_KEY_BASE);

        // Access tensor metadata shared memory
        tensor_meta_shmid = shmget(tensor_meta_key, sizeof(FileMetadataWithTensors), 0666);
        if (tensor_meta_shmid == -1) {
            throw std::runtime_error("Failed to find tensor metadata shared memory for " + filepath +
                                     " (key=" + std::to_string(tensor_meta_key) +
                                     "). Is tensor preloader running?");
        }

        // Attach to tensor metadata
        tensor_meta = shmat(tensor_meta_shmid, NULL, 0);
        if (tensor_meta == (void*)-1) {
            throw std::runtime_error("Failed to attach to tensor metadata shared memory for " + filepath);
        }

        // Get file tensor metadata
        FileMetadataWithTensors* file_tensor_meta = static_cast<FileMetadataWithTensors*>(tensor_meta);
        if (file_tensor_meta->preloaded_count == 0 || file_tensor_meta->tensor_data_size == 0) {
            shmdt(tensor_meta);
            throw std::runtime_error("No valid preloaded tensors found for " + filepath);
        }

        // Access tensor data
        tensor_data_shmid = shmget(tensor_data_key, file_tensor_meta->tensor_data_size, 0666);
        if (tensor_data_shmid == -1) {
            shmdt(tensor_meta);
            throw std::runtime_error("Failed to access tensor data segment for " + filepath +
                                    " (key=" + std::to_string(tensor_data_key) + ")");
        }

        tensor_data = shmat(tensor_data_shmid, NULL, 0);
        if (tensor_data == (void*)-1) {
            shmdt(tensor_meta);
            throw std::runtime_error("Failed to attach to tensor data for " + filepath);
        }

        has_preloaded_tensors = true;
        is_attached = true;

        log(DEBUG, "Successfully attached to shared memory for " + filepath + " with preloaded tensors");
    }

    ~SharedMemoryAccess() {
        if (is_attached) {
            if (tensor_data != nullptr) {
                shmdt(tensor_data);
            }
            if (tensor_meta != nullptr) {
                shmdt(tensor_meta);
            }
        }
    }

    // Removed get_data_slice method as it's no longer needed -
    // we only use preloaded tensors now

    // Check if there's a preloaded tensor for this offset
    bool has_preloaded_tensor(size_t offset) const {
        if (!has_preloaded_tensors || !is_attached) {
            return false;
        }

        // Get pointer to tensor metadata array (after the file metadata)
        FileMetadataWithTensors* file_tensor_meta = static_cast<FileMetadataWithTensors*>(tensor_meta);
        TensorMetadata* tensor_metadata = reinterpret_cast<TensorMetadata*>(file_tensor_meta + 1);

        // Search for tensor with matching offset
        for (int i = 0; i < file_tensor_meta->tensor_count; i++) {
            if (tensor_metadata[i].offset == static_cast<int64_t>(offset) && tensor_metadata[i].is_preloaded) {
                return true;
            }
        }

        return false;
    }

    // Get a preloaded tensor from shared memory
    torch::Tensor get_preloaded_tensor(size_t offset) const {
        if (!has_preloaded_tensors || !is_attached) {
            throw std::runtime_error("No preloaded tensors available");
        }

        // Get pointer to tensor metadata array (after the file metadata)
        FileMetadataWithTensors* file_tensor_meta = static_cast<FileMetadataWithTensors*>(tensor_meta);
        TensorMetadata* tensor_metadata = reinterpret_cast<TensorMetadata*>(file_tensor_meta + 1);

        // Find tensor with matching offset
        for (int i = 0; i < file_tensor_meta->tensor_count; i++) {
            if (tensor_metadata[i].offset == static_cast<int64_t>(offset) && tensor_metadata[i].is_preloaded) {
                // Get tensor shape
                std::vector<int64_t> dims;
                for (int d = 0; d < tensor_metadata[i].ndim; d++) {
                    dims.push_back(tensor_metadata[i].dims[d]);
                }

                // Create tensor options with the right dtype
                c10::ScalarType dtype = static_cast<c10::ScalarType>(tensor_metadata[i].dtype);
                auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

                // Get pointer to this specific tensor in shared memory
                // Use shm_offset to find the correct position in shared memory
                void* data_ptr = static_cast<char*>(tensor_data) + tensor_metadata[i].shm_offset;

                // Create a zero-copy tensor that references shared memory
                torch::Tensor tensor = torch::from_blob(
                    data_ptr,
                    dims,
                    [](void*) {}, // No-op deleter since we don't own the memory
                    options
                );

                return tensor;
            }
        }

        throw std::runtime_error("Tensor not found at offset " + std::to_string(offset));
    }

    // Check if we're successfully attached
    bool is_valid() const {
        return is_attached && tensor_meta != nullptr && tensor_data != nullptr;
    }

    // Check if this file has preloaded tensors
    bool has_tensors() const {
        return has_preloaded_tensors;
    }
};

// No longer needed - removed the tensor deserialization function since we require preloaded tensors

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

    // Use atomic for count and mutex for double values
    std::mutex timing_mutex;
    double total_data_access_time = 0.0;
    double total_tensor_load_time = 0.0;
    double total_tensor_copy_time = 0.0;

    // Track preloaded tensors
    std::atomic<size_t> preloaded_count(0);

    std::atomic<size_t> request_index(0);
    std::atomic<size_t> success_count(0);
    std::vector<std::thread> threads;

    // Create streams for each CUDA device
    std::map<int, std::vector<c10::cuda::CUDAStream>> device_streams;
    int streams_per_device = 4;  // Creating multiple streams per device

    // Find all unique CUDA devices
    std::set<int> cuda_devices;
    for (const auto& req : requests) {
        auto device = req.destination_tensor.device();
        if (device.is_cuda()) {
            cuda_devices.insert(device.index());
        }
    }

    // Create streams for each CUDA device
    for (int device_idx : cuda_devices) {
        // Store current device to restore later
        int prev_device = c10::cuda::current_device();

        // Set device for stream creation
        c10::cuda::set_device(device_idx);

        // Create multiple streams for this device
        for (int i = 0; i < streams_per_device; i++) {
            device_streams[device_idx].push_back(c10::cuda::getStreamFromPool(true)); // High priority stream
        }

        // Restore previous device
        c10::cuda::set_device(prev_device);

        log(INFO, "Created " + std::to_string(streams_per_device) +
                 " CUDA streams for device " + std::to_string(device_idx));
    }

    // Worker function that processes tensors
    auto process_tensors = [&](int thread_id) {
        // Get thread ID for logging
        std::stringstream ss;
        ss << std::this_thread::get_id();
        std::string thread_id_str = ss.str();

        // Thread local timing accumulators
        double thread_data_access = 0.0;
        double thread_tensor_load = 0.0;
        double thread_tensor_copy = 0.0;
        int processed_count = 0;
        int thread_preloaded = 0;

        // Cache for file accesses to avoid repeated connections
        std::map<std::string, std::shared_ptr<SharedMemoryAccess>> file_access_map;

        while (true) {
            // Get next request atomically
            size_t idx = request_index.fetch_add(1);
            if (idx >= requests.size()) {
                break;  // No more requests
            }

            TensorCopyRequest& req = requests[idx];
            auto start_time = std::chrono::high_resolution_clock::now();

            try {
                // STAGE 1: Access the file data from shared memory
                auto stage_start = std::chrono::high_resolution_clock::now();

                // Try to reuse existing access object for this file
                if (file_access_map.find(req.filepath) == file_access_map.end()) {
                    file_access_map[req.filepath] = std::make_shared<SharedMemoryAccess>(req.filepath);
                }

                auto& shm_access = file_access_map[req.filepath];

                auto stage_end = std::chrono::high_resolution_clock::now();
                double data_access_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    stage_end - stage_start
                ).count();
                thread_data_access += data_access_ms;

                // STAGE 2: Get the tensor from shared memory
                stage_start = std::chrono::high_resolution_clock::now();

                // Check if this file has preloaded tensors
                if (!shm_access->has_tensors() || !shm_access->has_preloaded_tensor(req.offset)) {
                    throw std::runtime_error("No preloaded tensor found at offset " +
                                           std::to_string(req.offset));
                }

                // Get preloaded tensor
                torch::Tensor cpu_tensor = shm_access->get_preloaded_tensor(req.offset);
                thread_preloaded++;

                stage_end = std::chrono::high_resolution_clock::now();
                double tensor_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    stage_end - stage_start
                ).count();
                thread_tensor_load += tensor_load_ms;

                // Verify tensor sizes match
                if (cpu_tensor.sizes() != req.destination_tensor.sizes()) {
                    throw std::runtime_error("Size mismatch for tensor: source " +
                                          std::to_string(cpu_tensor.numel()) +
                                          " vs destination " +
                                          std::to_string(req.destination_tensor.numel()));
                }

                // STAGE 3: Copy the tensor to its destination
                stage_start = std::chrono::high_resolution_clock::now();

                auto device = req.destination_tensor.device();
                if (device.is_cuda()) {
                    int device_idx = device.index();

                    // If we have streams for this device, use them
                    if (device_streams.find(device_idx) != device_streams.end()) {
                        // Round-robin stream selection
                        size_t stream_idx = idx % device_streams[device_idx].size();
                        c10::cuda::CUDAStream& stream = device_streams[device_idx][stream_idx];

                        // Store current device
                        int prev_device = c10::cuda::current_device();

                        // Set device
                        c10::cuda::set_device(device_idx);

                        // Set this stream as current and do a non-blocking copy
                        c10::cuda::setCurrentCUDAStream(stream);
                        req.destination_tensor.copy_(cpu_tensor, /*non_blocking=*/true);

                        // Restore previous device
                        c10::cuda::set_device(prev_device);
                    } else {
                        // Fallback to regular copy
                        req.destination_tensor.copy_(cpu_tensor);
                    }
                } else {
                    // For CPU tensors, just do a regular copy
                    req.destination_tensor.copy_(cpu_tensor);
                }

                stage_end = std::chrono::high_resolution_clock::now();
                double tensor_copy_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    stage_end - stage_start
                ).count();

                thread_tensor_copy += tensor_copy_ms;
                success_count++;
                processed_count++;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time
                ).count();

                // Detailed timing information for debugging performance
                if (idx % 100 == 0) { // Log every 100th tensor for less noise
                    log(INFO, "Tensor " + std::to_string(idx) + " timing: " +
                             "data_access=" + std::to_string(data_access_ms) + "ms, " +
                             "tensor_load=" + std::to_string(tensor_load_ms) + "ms, " +
                             "tensor_copy=" + std::to_string(tensor_copy_ms) + "ms, " +
                             "total=" + std::to_string(total_duration) + "ms");
                }
            } catch (const std::exception& e) {
                log(ERROR, "Thread " + thread_id_str + " error processing file " +
                          req.filepath + " at index " + std::to_string(idx) +
                          ": " + e.what());
            }

            // Log thread stats periodically
            if (processed_count >= 500) { // Every 500 tensors
                log(INFO, "Thread " + thread_id_str + " processed 500 tensors - " +
                          "avg times: data_access=" + std::to_string(thread_data_access/processed_count) + "ms, " +
                          "tensor_load=" + std::to_string(thread_tensor_load/processed_count) + "ms, " +
                          "tensor_copy=" + std::to_string(thread_tensor_copy/processed_count) + "ms");

                // Add thread stats to global counters with mutex protection
                {
                    std::lock_guard<std::mutex> lock(timing_mutex);
                    total_data_access_time += thread_data_access;
                    total_tensor_load_time += thread_tensor_load;
                    total_tensor_copy_time += thread_tensor_copy;
                    preloaded_count += thread_preloaded;
                }

                processed_count = 0;
                thread_data_access = 0;
                thread_tensor_load = 0;
                thread_tensor_copy = 0;
                thread_preloaded = 0;
            }
        }

        // Add any remaining stats before thread exits
        if (processed_count > 0) {
            std::lock_guard<std::mutex> lock(timing_mutex);
            total_data_access_time += thread_data_access;
            total_tensor_load_time += thread_tensor_load;
            total_tensor_copy_time += thread_tensor_copy;
            preloaded_count += thread_preloaded;
        }
    };

    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(process_tensors, i);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // For CUDA devices, synchronize to ensure all copies are complete
    for (int device_idx : cuda_devices) {
        // Store current device
        int prev_device = c10::cuda::current_device();

        // Set device
        c10::cuda::set_device(device_idx);

        // Synchronize this device (without arguments)
        c10::cuda::device_synchronize();

        // Restore previous device
        c10::cuda::set_device(prev_device);
    }

    // Calculate averages
    double avg_data_access = 0.0, avg_tensor_load = 0.0, avg_tensor_copy = 0.0;
    size_t success_count_value = success_count.load();
    if (success_count_value > 0) {
        avg_data_access = total_data_access_time / success_count_value;
        avg_tensor_load = total_tensor_load_time / success_count_value;
        avg_tensor_copy = total_tensor_copy_time / success_count_value;
    }

    // Report detailed completion stats
    log(INFO, "Completed processing " + std::to_string(success_count_value) +
             " out of " + std::to_string(request_count) + " tensors");

    // Report preloaded tensors count
    log(INFO, "Preloaded tensors used: " + std::to_string(preloaded_count.load()));

    log(INFO, "Performance breakdown (average time per tensor):");
    log(INFO, "  Data access:   " + std::to_string(avg_data_access) + "ms (" +
             std::to_string((avg_data_access / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(INFO, "  Tensor load:   " + std::to_string(avg_tensor_load) + "ms (" +
             std::to_string((avg_tensor_load / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(INFO, "  Tensor copy:   " + std::to_string(avg_tensor_copy) + "ms (" +
             std::to_string((avg_tensor_copy / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(INFO, "  Total:         " + std::to_string(avg_data_access + avg_tensor_load + avg_tensor_copy) + "ms");

    double total_time = total_data_access_time + total_tensor_load_time + total_tensor_copy_time;
    log(INFO, "Total time breakdown:");
    log(INFO, "  Data access:   " + std::to_string(total_data_access_time) + "ms (" +
             std::to_string((total_data_access_time / total_time) * 100) + "%)");
    log(INFO, "  Tensor load:   " + std::to_string(total_tensor_load_time) + "ms (" +
             std::to_string((total_tensor_load_time / total_time) * 100) + "%)");
    log(INFO, "  Tensor copy:   " + std::to_string(total_tensor_copy_time) + "ms (" +
             std::to_string((total_tensor_copy_time / total_time) * 100) + "%)");

    return success_count_value == request_count;
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

    // No longer exposing the removed function

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

    // Expose the set_log_level function
    m.def("set_log_level", &set_log_level,
          "Set the log level (DEBUG, INFO, WARNING, ERROR)",
          py::arg("level"));
}