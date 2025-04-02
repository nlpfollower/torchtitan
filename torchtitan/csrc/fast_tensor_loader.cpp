// fast_tensor_loader.cpp
#include "fast_tensor_loader.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <unistd.h>

// Function to narrow tensor according to offsets and lengths
torch::Tensor narrow_tensor_by_index(
    torch::Tensor tensor,
    const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& sizes
) {
    torch::Tensor narrowed_tensor = tensor;

    // Make sure we have valid offsets and sizes
    if (offsets.size() != sizes.size()) {
        throw std::runtime_error("Offsets and sizes must have the same length");
    }

    // Apply narrowing for each dimension where size is smaller than tensor size
    for (size_t idx = 0; idx < offsets.size() && idx < tensor.dim(); idx++) {
        if (sizes[idx] < tensor.size(idx)) {
            narrowed_tensor = narrowed_tensor.narrow(idx, offsets[idx], sizes[idx]);
        }
    }

    return narrowed_tensor;
}

// SharedMemoryAccess implementation
SharedMemoryAccess::SharedMemoryAccess(const std::string& path)
    : filepath(path),
      tensor_meta_shmid(-1), tensor_data_shmid(-1),
      tensor_meta(nullptr), tensor_data(nullptr),
      is_attached(false), has_preloaded_tensors(false)
{
    // Generate keys based on filepath
    int tensor_meta_key = generate_file_key(filepath, SHM_TENSOR_META_KEY_BASE);
    int tensor_data_key = generate_file_key(filepath, SHM_TENSOR_KEY_BASE);

    // Access tensor metadata shared memory
    tensor_meta_shmid = shmget(tensor_meta_key, sizeof(FileMetadata), 0666);
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
    FileMetadata* file_tensor_meta = static_cast<FileMetadata*>(tensor_meta);
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

SharedMemoryAccess::~SharedMemoryAccess() {
    if (is_attached) {
        if (tensor_data != nullptr) {
            shmdt(tensor_data);
        }
        if (tensor_meta != nullptr) {
            shmdt(tensor_meta);
        }
    }
}

// Check if there's a preloaded tensor for this offset
bool SharedMemoryAccess::has_preloaded_tensor(size_t offset) const {
    if (!has_preloaded_tensors || !is_attached) {
        return false;
    }

    // Get pointer to tensor metadata array (after the file metadata)
    FileMetadata* file_tensor_meta = static_cast<FileMetadata*>(tensor_meta);
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
torch::Tensor SharedMemoryAccess::get_preloaded_tensor(size_t offset) const {
    if (!has_preloaded_tensors || !is_attached) {
        throw std::runtime_error("No preloaded tensors available");
    }

    // Get pointer to tensor metadata array (after the file metadata)
    FileMetadata* file_tensor_meta = static_cast<FileMetadata*>(tensor_meta);
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
bool SharedMemoryAccess::is_valid() const {
    return is_attached && tensor_meta != nullptr && tensor_data != nullptr;
}

// Check if this file has preloaded tensors
bool SharedMemoryAccess::has_tensors() const {
    return has_preloaded_tensors;
}

// Function that loads and copies tensors directly to their destination
bool load_and_copy_tensors_parallel(
    std::vector<TensorCopyRequest>& requests,
    int num_threads = -1,
    int streams_per_device = 4
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

        std::string rank_prefix = "rank" + std::to_string(thread_id) + "]:";

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

                // Apply narrowing based on tensor_offsets and tensor_lengths
                if (!req.tensor_offsets.empty() && !req.tensor_lengths.empty()) {
                    try {
                        cpu_tensor = narrow_tensor_by_index(cpu_tensor, req.tensor_offsets, req.tensor_lengths);
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Failed to narrow tensor: " + std::string(e.what()));
                    }
                }

                thread_preloaded++;

                stage_end = std::chrono::high_resolution_clock::now();
                double tensor_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    stage_end - stage_start
                ).count();
                thread_tensor_load += tensor_load_ms;

                // Verify tensor sizes match after narrowing
                if (cpu_tensor.sizes() != req.destination_tensor.sizes()) {
                    std::stringstream ss;
                    ss << "Size mismatch for tensor: source [";
                    for (int i = 0; i < cpu_tensor.dim(); i++) {
                        ss << cpu_tensor.size(i);
                        if (i < cpu_tensor.dim() - 1) ss << ", ";
                    }
                    ss << "] vs destination [";
                    for (int i = 0; i < req.destination_tensor.dim(); i++) {
                        ss << req.destination_tensor.size(i);
                        if (i < req.destination_tensor.dim() - 1) ss << ", ";
                    }
                    ss << "]";

                    throw std::runtime_error(ss.str());
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
                        cpu_tensor = cpu_tensor.pin_memory();
                        req.destination_tensor.copy_(cpu_tensor, /*non_blocking=*/true);

                        // Restore previous device
                        c10::cuda::set_device(prev_device);
                    } else {
                        throw std::runtime_error("CUDA streams not available for device " + std::to_string(device_idx));
                    }
                } else {
                    // For CPU tensors, just do a regular copy
                    throw std::runtime_error("CPU tensor copy not implemented yet");
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

                // Calculate tensor size in MB for transfer rate
                double tensor_size_mb = (cpu_tensor.nbytes() / (1024.0 * 1024.0));

                // Calculate transfer rate in MB/s
                double transfer_rate_mbps = 0.0;
                if (tensor_copy_ms > 0) {
                    transfer_rate_mbps = tensor_size_mb / (tensor_copy_ms / 1000.0);
                }

                // Detailed timing information for debugging performance
                // Log every tensor for more detailed analysis
                log(DEBUG, rank_prefix + "[PRELOADER][INFO] Tensor " + std::to_string(idx) +
                         " (size=" + std::to_string(tensor_size_mb) + " MB) timing: " +
                         "data_access=" + std::to_string(data_access_ms) + "ms, " +
                         "tensor_load=" + std::to_string(tensor_load_ms) + "ms, " +
                         "tensor_copy=" + std::to_string(tensor_copy_ms) + "ms " +
                         "(rate=" + std::to_string(transfer_rate_mbps) + " MB/s), " +
                         "total=" + std::to_string(total_duration) + "ms");
            } catch (const std::exception& e) {
                log(ERROR, rank_prefix + "[PRELOADER][ERROR] Thread " + thread_id_str + " error processing file " +
                          req.filepath + " at index " + std::to_string(idx) +
                          ": " + e.what());
            }

            // Log thread stats periodically
            if (processed_count >= 500) { // Every 500 tensors
                log(DEBUG, rank_prefix + "[PRELOADER][INFO] Thread " + thread_id_str + " processed 500 tensors - " +
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

        // Synchronize this device
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
    log(DEBUG, "[PRELOADER][SUMMARY] Completed processing " + std::to_string(success_count_value) +
             " out of " + std::to_string(request_count) + " tensors");

    // Report preloaded tensors count
    log(DEBUG, "[PRELOADER][SUMMARY] Preloaded tensors used: " + std::to_string(preloaded_count.load()));

    log(DEBUG, "[PRELOADER][SUMMARY] Performance breakdown (average time per tensor):");
    log(DEBUG, "[PRELOADER][SUMMARY]   Data access:   " + std::to_string(avg_data_access) + "ms (" +
             std::to_string((avg_data_access / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(DEBUG, "[PRELOADER][SUMMARY]   Tensor load:   " + std::to_string(avg_tensor_load) + "ms (" +
             std::to_string((avg_tensor_load / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(DEBUG, "[PRELOADER][SUMMARY]   Tensor copy:   " + std::to_string(avg_tensor_copy) + "ms (" +
             std::to_string((avg_tensor_copy / (avg_data_access + avg_tensor_load + avg_tensor_copy)) * 100) + "%)");
    log(DEBUG, "[PRELOADER][SUMMARY]   Total:         " + std::to_string(avg_data_access + avg_tensor_load + avg_tensor_copy) + "ms");

    double total_time = total_data_access_time + total_tensor_load_time + total_tensor_copy_time;
    log(DEBUG, "[PRELOADER][SUMMARY] Total time breakdown:");
    log(DEBUG, "[PRELOADER][SUMMARY]   Data access:   " + std::to_string(total_data_access_time) + "ms (" +
             std::to_string((total_data_access_time / total_time) * 100) + "%)");
    log(DEBUG, "[PRELOADER][SUMMARY]   Tensor load:   " + std::to_string(total_tensor_load_time) + "ms (" +
             std::to_string((total_tensor_load_time / total_time) * 100) + "%)");
    log(DEBUG, "[PRELOADER][SUMMARY]   Tensor copy:   " + std::to_string(total_tensor_copy_time) + "ms (" +
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

    py::class_<TensorCopyRequest>(m, "TensorCopyRequest")
        .def(py::init<>())
        .def_readwrite("filepath", &TensorCopyRequest::filepath)
        .def_readwrite("offset", &TensorCopyRequest::offset)
        .def_readwrite("length", &TensorCopyRequest::length)
        .def_readwrite("index", &TensorCopyRequest::index)
        .def_readwrite("tensor_offsets", &TensorCopyRequest::tensor_offsets)
        .def_readwrite("tensor_lengths", &TensorCopyRequest::tensor_lengths)
        .def_readwrite("destination_tensor", &TensorCopyRequest::destination_tensor);

    m.def("load_and_copy_tensors_parallel",
      static_cast<bool (*)(std::vector<TensorCopyRequest>&, int, int)>(&load_and_copy_tensors_parallel),
      "Load and directly copy multiple tensors to their destinations in parallel",
      py::arg("requests"),
      py::arg("num_threads") = -1,
      py::arg("streams_per_device") = 4);

    // Expose the set_log_level function
    m.def("set_log_level", &set_log_level,
          "Set the log level (DEBUG, INFO, WARNING, ERROR)",
          py::arg("level"));
}