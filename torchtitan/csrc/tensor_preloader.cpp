#include "gloo_file_broadcast.h"
#include "tensor_preloader.h"
#include <chrono>
#include <atomic>
#include <stdexcept>

// Global statistics
struct PreloadStats {
    std::atomic<int> total_count{0};
    std::atomic<int> preloaded_count{0};
    std::atomic<size_t> total_memory{0};
    std::atomic<uint64_t> read_time_ns{0};
    std::atomic<uint64_t> deserialize_time_ns{0};
    std::atomic<uint64_t> copy_time_ns{0};
} g_stats;

// Helper function to atomically add seconds
inline void atomic_add_seconds(std::atomic<uint64_t>& counter, double seconds) {
    uint64_t ns = static_cast<uint64_t>(seconds * 1e9);
    counter.fetch_add(ns, std::memory_order_relaxed);
}

// Primary tensor loading function - broken into manageable steps
bool preload_file_tensors(const std::string& filepath,
                        const std::vector<py::dict>& tensor_infos,
                        int num_threads,
                        GlooFileBroadcast* broadcaster = nullptr) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Initial setup and parameter extraction
    int tensor_count = tensor_infos.size();
    g_stats.total_count += tensor_count;

    if (tensor_count == 0) {
        log(INFO, "No tensors to preload for " + filepath);
        return true;
    }

    log(INFO, "Preloading " + std::to_string(tensor_count) + " tensors from " + filepath);

    // 2. Generate shared memory keys and clean up existing segments
    int tensor_meta_key = generate_file_key(filepath, SHM_TENSOR_META_KEY_BASE);
    int tensor_data_key = generate_file_key(filepath, SHM_TENSOR_KEY_BASE);

    SharedMemory::cleanup_file_segments(filepath);

    // 3. Memory map the file
    MemoryMappedFile mapped_file(filepath, broadcaster);
    if (!mapped_file.valid()) {
        throw std::runtime_error("Failed to memory map file: " + filepath);
    }

    // 4. Determine thread count and initial shared memory size
    int worker_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();
    worker_threads = std::min(worker_threads, std::max(1, tensor_count / 10 + 1));

    log(INFO, "Using " + std::to_string(worker_threads) + " worker threads for tensor preloading");

    // 5. Extract tensor information
    std::vector<int64_t> tensor_offsets(tensor_count);
    std::vector<int64_t> tensor_lengths(tensor_count);
    std::vector<std::string> tensor_keys(tensor_count);

    for (int i = 0; i < tensor_count; i++) {
        py::dict info = tensor_infos[i];
        tensor_offsets[i] = py::cast<int64_t>(info["offset"]);
        tensor_lengths[i] = py::cast<int64_t>(info["length"]);
        tensor_keys[i] = py::cast<std::string>(info["key"]);
    }

    // 6. Create shared memory for tensor metadata
    size_t metadata_size = sizeof(TensorMetadata) * tensor_count + sizeof(FileMetadata);
    int tensor_meta_shmid = SharedMemory::create_segment(tensor_meta_key, metadata_size);

    if (tensor_meta_shmid == -1) {
        throw std::runtime_error("Failed to create tensor metadata shared memory");
    }

    // 7. Attach to tensor metadata segment
    void* shmaddr = SharedMemory::attach_segment(tensor_meta_shmid);
    if (shmaddr == (void*)-1) {
        throw std::runtime_error("Failed to attach to tensor metadata shared memory");
    }

    // 8. Initialize metadata structures
    FileMetadata* file_metadata = static_cast<FileMetadata*>(shmaddr);
    TensorMetadata* tensor_metadata = reinterpret_cast<TensorMetadata*>(file_metadata + 1);

    file_metadata->tensor_count = tensor_count;
    file_metadata->preloaded_count = 0;
    file_metadata->tensor_data_size = 0;

    for (int i = 0; i < tensor_count; i++) {
        tensor_metadata[i].offset = tensor_offsets[i];
        tensor_metadata[i].length = tensor_lengths[i];
        tensor_metadata[i].is_preloaded = false;
        strncpy(tensor_metadata[i].key, tensor_keys[i].c_str(), sizeof(tensor_metadata[i].key) - 1);
        tensor_metadata[i].key[sizeof(tensor_metadata[i].key) - 1] = '\0';
    }

    // 9. Estimate initial shared memory size (use file size with buffer as rough estimate)
    size_t file_size = mapped_file.size();
    size_t estimated_size = file_size * 1.5; // 50% buffer as a rough estimate

    // Ensure a minimum size (e.g., 10MB) and cap at a maximum reasonable size
    estimated_size = std::max(estimated_size, static_cast<size_t>(10 * 1024 * 1024));

    log(INFO, "Allocating initial shared memory segment of " +
             std::to_string(estimated_size / (1024 * 1024)) + "MB for tensor data");

    // 10. Create shared memory for tensor data
    bool use_huge_pages = (estimated_size >= 1024 * 1024 * 1024); // 1GB threshold
    int tensor_data_shmid = SharedMemory::create_segment(tensor_data_key, estimated_size, use_huge_pages);

    if (tensor_data_shmid == -1) {
        SharedMemory::detach_segment(shmaddr);
        throw std::runtime_error("Failed to create tensor data shared memory: " + std::to_string(errno));
    }

    // 11. Attach to tensor data segment
    void* tensor_data = SharedMemory::attach_segment(tensor_data_shmid);
    if (tensor_data == (void*)-1) {
        SharedMemory::detach_segment(shmaddr);
        throw std::runtime_error("Failed to attach to tensor data shared memory");
    }

    // 12. Setup for single-pass loading
    ThreadPool pool(worker_threads);
    std::atomic<int> success_count(0);
    std::mutex offset_mutex; // Only for offset calculation
    std::atomic<size_t> current_offset(0);

    // Add a completion barrier to ensure all threads finish before resources are released
    std::atomic<int> active_tasks(0);
    std::condition_variable completion_cv;
    std::mutex completion_mutex;
    std::vector<std::exception_ptr> thread_exceptions;
    std::mutex exceptions_mutex;

    // 13. Process tensors in a single pass
    std::vector<std::future<void>> futures;
    for (int i = 0; i < tensor_count; i++) {
        active_tasks.fetch_add(1);
        futures.push_back(pool.enqueue([&, i]() {
            auto task_start = std::chrono::high_resolution_clock::now();
            bool task_succeeded = false;

            try {
                // Get tensor data from memory-mapped file
                auto read_start = std::chrono::high_resolution_clock::now();
                const char* tensor_data_ptr = mapped_file.get_tensor_data(tensor_offsets[i], tensor_lengths[i]);
                auto read_end = std::chrono::high_resolution_clock::now();
                double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
                atomic_add_seconds(g_stats.read_time_ns, read_ms / 1000.0);

                if (!tensor_data_ptr) {
                    throw std::runtime_error("Failed to access tensor data for tensor " + std::to_string(i));
                }

                // CONCURRENT PART: Deserialize the tensor
                auto deserialize_start = std::chrono::high_resolution_clock::now();
                std::vector<char> data_vec(tensor_data_ptr, tensor_data_ptr + tensor_lengths[i]);
                torch::Tensor tensor;

                try {
                    auto ivalue = torch::pickle_load(data_vec);
                    if (ivalue.isTensor()) {
                        tensor = ivalue.toTensor().cpu();
                    } else {
                        throw std::runtime_error("Deserialized data is not a tensor for tensor " + std::to_string(i));
                    }
                } catch (const c10::Error& e) {
                    throw std::runtime_error("PyTorch error deserializing tensor " + std::to_string(i) +
                            ": " + e.what());
                }

                auto deserialize_end = std::chrono::high_resolution_clock::now();
                double deserialize_ms = std::chrono::duration<double, std::milli>(deserialize_end - deserialize_start).count();
                atomic_add_seconds(g_stats.deserialize_time_ns, deserialize_ms / 1000.0);

                if (!tensor.defined()) {
                    throw std::runtime_error("Failed to create tensor from deserialized data for tensor " + std::to_string(i));
                }

                // Make sure tensor is contiguous
                if (!tensor.is_contiguous()) {
                    tensor = tensor.contiguous();
                }

                // Calculate tensor size
                size_t tensor_size = tensor.nbytes();

                // Update metadata (thread-safe part)
                {
                    tensor_metadata[i].dtype = static_cast<int64_t>(tensor.scalar_type());
                    tensor_metadata[i].ndim = tensor.dim();
                    for (int64_t d = 0; d < tensor.dim() && d < 8; d++) {
                        tensor_metadata[i].dims[d] = tensor.size(d);
                    }
                }

                // MINIMIZED LOCK: Only calculate offset under lock
                size_t offset;
                {
                    std::lock_guard<std::mutex> lock(offset_mutex);

                    // Check if we have enough space left
                    size_t required_offset = current_offset.load() + tensor_size;
                    if (required_offset > estimated_size) {
                        throw std::runtime_error("Not enough shared memory for tensor " + std::to_string(i) +
                                 ". Consider using a larger initial allocation.");
                    }

                    // Get current offset and increment atomically
                    offset = current_offset.load();

                    // Update offsets - this is all we need the lock for
                    current_offset.fetch_add(tensor_size);

                    // Update tensor metadata that depends on offset
                    tensor_metadata[i].shm_offset = offset;
                }

                // PARALLEL COPY: Actually copy the data after releasing the lock
                auto copy_start = std::chrono::high_resolution_clock::now();

                // Access tensor_data safely - it's a shared resource
                if (tensor_data) {
                    // Copy tensor data to shared memory (no lock needed)
                    memcpy(static_cast<char*>(tensor_data) + offset, tensor.data_ptr(), tensor_size);

                    // Mark as preloaded - no lock needed since we're only writing to our assigned slot
                    tensor_metadata[i].is_preloaded = true;
                    success_count.fetch_add(1);
                    task_succeeded = true;
                } else {
                    throw std::runtime_error("Tensor data pointer is null for tensor " + std::to_string(i));
                }

                auto copy_end = std::chrono::high_resolution_clock::now();
                double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
                atomic_add_seconds(g_stats.copy_time_ns, copy_ms / 1000.0);

                // Log slow processing
                auto task_end = std::chrono::high_resolution_clock::now();
                double task_ms = std::chrono::duration<double, std::milli>(task_end - task_start).count();
                if (task_ms > 1000) {
                    log(WARNING, "Slow tensor processing for tensor " + std::to_string(i) +
                               ": " + std::to_string(task_ms) + "ms");
                }
            } catch (...) {
                // Capture the exception to be rethrown later
                std::lock_guard<std::mutex> lock(exceptions_mutex);
                thread_exceptions.push_back(std::current_exception());
            }

            // Notify that this task is complete
            int remaining = active_tasks.fetch_sub(1) - 1;
            if (remaining == 0) {
                // This was the last task
                std::unique_lock<std::mutex> lock(completion_mutex);
                completion_cv.notify_all();
            }
        }));

        // Log progress periodically
        if ((i+1) % 100 == 0 || (i+1) == tensor_count) {
            log(INFO, "Queued tensor processing: " + std::to_string(i+1) + "/" +
                     std::to_string(tensor_count) + " tensors");
        }
    }

    // 14. Wait for all tasks to complete and track progress
    {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [&]() { return active_tasks.load() == 0; });
    }

    // Check if any thread encountered an exception
    {
        std::lock_guard<std::mutex> lock(exceptions_mutex);
        if (!thread_exceptions.empty()) {
            // Clean up resources before throwing
            SharedMemory::detach_segment(shmaddr);
            SharedMemory::detach_segment(tensor_data);

            // Rethrow the first exception
            std::rethrow_exception(thread_exceptions[0]);
        }
    }

    // Now safe to check results and clean up
    int total_success = success_count.load();

    // 15. Update final metadata
    file_metadata->preloaded_count = total_success;
    file_metadata->tensor_data_size = current_offset.load();

    // 16. Update global stats
    g_stats.preloaded_count += total_success;
    g_stats.total_memory += current_offset.load();

    // 17. Clean up resources
    SharedMemory::detach_segment(shmaddr);
    SharedMemory::detach_segment(tensor_data);

    // 18. Report results
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    log(INFO, "Successfully preloaded " + std::to_string(total_success) +
             " of " + std::to_string(tensor_count) + " tensors from " + filepath +
             " (" + std::to_string(current_offset.load() / (1024 * 1024)) + "MB) in " +
             std::to_string(elapsed_seconds) + " seconds (" +
             std::to_string(total_success > 0 ? total_success / elapsed_seconds : 0) + " tensors/sec)");

    if (total_success == 0) {
        throw std::runtime_error("Failed to preload any tensors from " + filepath);
    }

    return true;
}

// Global preloader function for multiple files
bool preload_tensors(py::list file_tensors, int num_threads,
                    int rank = 0, int world_size = 1,
                    const std::string& redis_host = "localhost",
                    int redis_port = 6379,
                    const std::string& run_id = "") {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Reset global statistics
    g_stats.total_count = 0;
    g_stats.preloaded_count = 0;
    g_stats.total_memory = 0;
    g_stats.read_time_ns = 0;
    g_stats.deserialize_time_ns = 0;
    g_stats.copy_time_ns = 0;

    log(INFO, "Starting tensor preloading for " +
             std::to_string(file_tensors.size()) + " files");

    std::vector<std::string> failed_files;
    std::unique_ptr<GlooFileBroadcast> broadcaster;
    if (world_size > 1) {
        log(INFO, "Initializing distributed mode with rank " + std::to_string(rank) +
                 " of " + std::to_string(world_size));

        broadcaster = std::make_unique<GlooFileBroadcast>(
            rank, world_size, redis_host, redis_port, run_id);

        if (!broadcaster->initialize()) {
            throw std::runtime_error("Failed to initialize GlooFileBroadcast");
        }
    }

    // Process each file
    for (auto& item : file_tensors) {
        py::dict file_entry = py::cast<py::dict>(item);
        std::string filepath = py::cast<std::string>(file_entry["filepath"]);
        py::list tensors = py::cast<py::list>(file_entry["tensors"]);

        // Convert tensors metadata to vector of dicts
        std::vector<py::dict> tensor_infos;
        for (auto& tensor_obj : tensors) {
            tensor_infos.push_back(py::cast<py::dict>(tensor_obj));
        }

        // Determine thread count for this file - scale based on tensor count
        int file_threads = num_threads;
        if (file_threads <= 0) {
            file_threads = std::thread::hardware_concurrency();
        }

        // Limit threads for small files
        int tensor_count = tensor_infos.size();
        if (tensor_count < 100) {
            file_threads = std::min(file_threads, std::max(1, tensor_count / 10));
        }

        // Preload tensors for this file
        try {
            preload_file_tensors(filepath, tensor_infos, file_threads, broadcaster.get());
        } catch (const std::exception& e) {
            // Collect the failed file and propagate the exception
            failed_files.push_back(filepath);
            throw std::runtime_error("Failed to preload tensors for " + filepath + ": " + e.what());
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    log(INFO, "Completed tensor preloading: " +
             std::to_string(g_stats.preloaded_count) + "/" +
             std::to_string(g_stats.total_count) + " tensors, " +
             std::to_string(g_stats.total_memory / (1024.0 * 1024.0 * 1024.0)) +
             "GB shared memory in " + std::to_string(elapsed_seconds) + " seconds");

    // Performance breakdown
    log(INFO, "Performance breakdown:");

    double read_seconds = g_stats.read_time_ns.load() / 1e9;
    double deserialize_seconds = g_stats.deserialize_time_ns.load() / 1e9;
    double copy_seconds = g_stats.copy_time_ns.load() / 1e9;

    log(INFO, "  Read time: " + std::to_string(read_seconds) + " seconds");
    log(INFO, "  Deserialize time: " + std::to_string(deserialize_seconds) + " seconds");
    log(INFO, "  Copy time: " + std::to_string(copy_seconds) + " seconds");

    if (g_stats.preloaded_count > 0) {
        double avg_tensor_time = elapsed_seconds / g_stats.preloaded_count;
        log(INFO, "Average time per tensor: " + std::to_string(avg_tensor_time * 1000) + " ms");
    }

    if (g_stats.preloaded_count == 0) {
        throw std::runtime_error("Failed to preload any tensors");
    }

    return true;
}

// Get preloading statistics as a Python dictionary
py::dict get_preload_stats() {
    py::dict stats;
    stats["total_count"] = g_stats.total_count.load();
    stats["preloaded_count"] = g_stats.preloaded_count.load();
    stats["memory_bytes"] = g_stats.total_memory.load();
    stats["memory_gb"] = g_stats.total_memory.load() / (1024.0 * 1024.0 * 1024.0);
    stats["read_time"] = g_stats.read_time_ns.load() / 1e9; // Convert to seconds
    stats["deserialize_time"] = g_stats.deserialize_time_ns.load() / 1e9;
    stats["copy_time"] = g_stats.copy_time_ns.load() / 1e9;
    return stats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Main function to preload tensors
    m.def("preload_tensors",
          [](py::list file_tensors, int num_threads,
             int rank, int world_size,
             const std::string& redis_host, int redis_port,
             const std::string& run_id) {
              return preload_tensors(file_tensors, num_threads,
                                    rank, world_size,
                                    redis_host, redis_port, run_id);
          },
          "Preload tensors from checkpoint files into shared memory with optional broadcast",
          py::arg("file_tensors"),
          py::arg("num_threads") = 0,
          py::arg("rank") = 0,
          py::arg("world_size") = 1,
          py::arg("redis_host") = "localhost",
          py::arg("redis_port") = 6379,
          py::arg("run_id") = "");

    // Get preloading statistics
    m.def("get_preload_stats", &get_preload_stats,
          "Get tensor preloading statistics");

    // Set log level
    m.def("set_log_level", &set_log_level,
          "Set the log level",
          py::arg("level"));

    // Clean up shared memory
    m.def("cleanup_shared_memory", &SharedMemory::cleanup_all_segments,
          "Clean up all shared memory segments created by this module");
}