#include "tensor_preloader.h"
#include <chrono>
#include <atomic>

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

// Function to analyze tensor and get its size
size_t analyze_tensor_size(const char* tensor_data, size_t length, TensorMetadata& metadata) {
    if (tensor_data == nullptr || length == 0) {
        return 0;
    }

    try {
        // Create a copy with pre-allocated memory for pickle_load
        std::vector<char> data_vec(tensor_data, tensor_data + length);

        // Deserialize the tensor
        auto ivalue = torch::pickle_load(data_vec);
        if (!ivalue.isTensor()) {
            return 0;
        }

        torch::Tensor tensor = ivalue.toTensor().cpu();
        if (!tensor.defined()) {
            return 0;
        }

        // Make contiguous if needed
        if (!tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }

        // Update metadata
        metadata.dtype = static_cast<int64_t>(tensor.scalar_type());
        metadata.ndim = tensor.dim();
        for (int64_t d = 0; d < tensor.dim() && d < 8; d++) {
            metadata.dims[d] = tensor.size(d);
        }

        // Return tensor size in bytes
        return tensor.nbytes();
    } catch (const c10::Error& e) {
        log(ERROR, "PyTorch error analyzing tensor: " + std::string(e.what()));
    } catch (const std::exception& e) {
        log(ERROR, "Exception analyzing tensor: " + std::string(e.what()));
    } catch (...) {
        log(ERROR, "Unknown error analyzing tensor");
    }

    return 0;
}

// Function to load tensor into shared memory
bool load_tensor_to_shared_memory(const char* tensor_data, size_t length,
                                void* shm_data, size_t shm_offset) {
    try {
        // Create a vector for pickle_load
        std::vector<char> data_vec(tensor_data, tensor_data + length);

        // Deserialize the tensor
        auto ivalue = torch::pickle_load(data_vec);
        if (!ivalue.isTensor()) {
            return false;
        }

        torch::Tensor tensor = ivalue.toTensor().cpu();
        if (!tensor.defined()) {
            return false;
        }

        // Make contiguous if needed
        if (!tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }

        // Copy to shared memory
        memcpy(static_cast<char*>(shm_data) + shm_offset, tensor.data_ptr(), tensor.nbytes());

        return true;
    } catch (const c10::Error& e) {
        log(ERROR, "PyTorch error loading tensor: " + std::string(e.what()));
    } catch (const std::exception& e) {
        log(ERROR, "Exception loading tensor: " + std::string(e.what()));
    } catch (...) {
        log(ERROR, "Unknown error loading tensor");
    }

    return false;
}

// Primary tensor loading function - broken into manageable steps
bool preload_file_tensors(const std::string& filepath,
                        const std::vector<py::dict>& tensor_infos,
                        int num_threads) {
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
    MemoryMappedFile mapped_file(filepath);
    if (!mapped_file.valid()) {
        log(ERROR, "Failed to memory map file: " + filepath);
        return false;
    }

    // 4. Determine thread count
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
        log(ERROR, "Failed to create tensor metadata shared memory");
        return false;
    }

    // 7. Attach to tensor metadata segment
    void* shmaddr = SharedMemory::attach_segment(tensor_meta_shmid);
    if (shmaddr == (void*)-1) {
        log(ERROR, "Failed to attach to tensor metadata shared memory");
        return false;
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

    // 9. Create and configure thread pool
    ThreadPool pool(worker_threads);
    std::mutex tensor_sizes_mutex;
    std::atomic<size_t> total_tensor_size(0);
    std::vector<size_t> tensor_sizes(tensor_count, 0);

    // === FIRST PASS: Calculate tensor sizes ===
    std::vector<std::future<size_t>> size_futures;

    for (int i = 0; i < tensor_count; i++) {
        size_futures.push_back(pool.enqueue([&, i]() {
            auto task_start = std::chrono::high_resolution_clock::now();

            // Get tensor data
            auto read_start = std::chrono::high_resolution_clock::now();
            const char* tensor_data = mapped_file.get_tensor_data(tensor_offsets[i], tensor_lengths[i]);
            auto read_end = std::chrono::high_resolution_clock::now();
            double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
            atomic_add_seconds(g_stats.read_time_ns, read_ms / 1000.0);

            if (!tensor_data) {
                return size_t(0);
            }

            // Calculate tensor size
            auto deserialize_start = std::chrono::high_resolution_clock::now();
            size_t tensor_size = analyze_tensor_size(tensor_data, tensor_lengths[i], tensor_metadata[i]);
            auto deserialize_end = std::chrono::high_resolution_clock::now();
            double deserialize_ms = std::chrono::duration<double, std::milli>(deserialize_end - deserialize_start).count();
            atomic_add_seconds(g_stats.deserialize_time_ns, deserialize_ms / 1000.0);

            // Log slow processing
            auto task_end = std::chrono::high_resolution_clock::now();
            double task_ms = std::chrono::duration<double, std::milli>(task_end - task_start).count();
            if (task_ms > 1000) {
                log(WARNING, "Slow tensor size calculation for tensor " + std::to_string(i) +
                           ": " + std::to_string(task_ms) + "ms");
            }

            return tensor_size;
        }));
    }

    // Wait for all size calculations and collect results
    for (int i = 0; i < tensor_count; i++) {
        tensor_sizes[i] = size_futures[i].get();
        if (tensor_sizes[i] > 0) {
            total_tensor_size.fetch_add(tensor_sizes[i]);
        }

        if ((i+1) % 100 == 0 || (i+1) == tensor_count) {
            log(INFO, "Size calculation progress: " + std::to_string(i+1) + "/" +
                     std::to_string(tensor_count) + " tensors");
        }
    }

    // 10. Calculate final size with safety buffer
    if (total_tensor_size.load() == 0) {
        log(ERROR, "No valid tensors found in " + filepath);
        SharedMemory::detach_segment(shmaddr);
        return false;
    }

    size_t final_size = total_tensor_size.load() + (total_tensor_size.load() / 50); // 2% buffer

    log(INFO, "Size calculation complete. Total tensor data size: " +
             std::to_string(final_size / (1024 * 1024)) + "MB");

    // 11. Create shared memory for tensor data
    log(INFO, "Creating shared memory segment of " +
             std::to_string(final_size / (1024 * 1024)) + "MB for tensor data");

    // Use huge pages for large allocations
    bool use_huge_pages = (final_size >= 1024 * 1024 * 1024); // 1GB threshold
    int tensor_data_shmid = SharedMemory::create_segment(tensor_data_key, final_size, use_huge_pages);

    if (tensor_data_shmid == -1) {
        log(ERROR, "Failed to create tensor data shared memory: " +
                 std::to_string(errno));
        SharedMemory::detach_segment(shmaddr);
        return false;
    }

    // 12. Attach to tensor data segment
    void* tensor_data = SharedMemory::attach_segment(tensor_data_shmid);
    if (tensor_data == (void*)-1) {
        log(ERROR, "Failed to attach to tensor data shared memory");
        SharedMemory::detach_segment(shmaddr);
        return false;
    }

    // 13. Second pass: Load tensors into shared memory
    std::atomic<size_t> current_offset(0);
    std::atomic<int> success_count(0);
    std::vector<std::future<bool>> load_futures;

    for (int i = 0; i < tensor_count; i++) {
        // Skip tensors with zero size (failed analysis)
        if (tensor_sizes[i] == 0) {
            continue;
        }

        // Reserve space in shared memory (atomic operation)
        size_t offset = current_offset.fetch_add(tensor_sizes[i]);

        // Check if we have enough space
        if (offset + tensor_sizes[i] > final_size) {
            log(ERROR, "Not enough shared memory for tensor " + std::to_string(i));
            continue;
        }

        // Update metadata with offset
        {
            std::lock_guard<std::mutex> lock(tensor_sizes_mutex);
            tensor_metadata[i].shm_offset = offset;
        }

        // Queue loading task
        load_futures.push_back(pool.enqueue([&, i, offset]() {
            auto task_start = std::chrono::high_resolution_clock::now();

            // Get tensor data from memory-mapped file
            auto read_start = std::chrono::high_resolution_clock::now();
            const char* tensor_data_ptr = mapped_file.get_tensor_data(tensor_offsets[i], tensor_lengths[i]);
            auto read_end = std::chrono::high_resolution_clock::now();
            double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
            atomic_add_seconds(g_stats.read_time_ns, read_ms / 1000.0);

            if (!tensor_data_ptr) {
                return false;
            }

            // Load tensor into shared memory
            auto deserialize_start = std::chrono::high_resolution_clock::now();
            bool success = load_tensor_to_shared_memory(
                tensor_data_ptr, tensor_lengths[i],
                tensor_data, offset
            );
            auto deserialize_end = std::chrono::high_resolution_clock::now();
            double deserialize_ms = std::chrono::duration<double, std::milli>(deserialize_end - deserialize_start).count();
            atomic_add_seconds(g_stats.deserialize_time_ns, deserialize_ms / 1000.0);

            if (success) {
                // Update metadata to mark as preloaded
                {
                    std::lock_guard<std::mutex> lock(tensor_sizes_mutex);
                    tensor_metadata[i].is_preloaded = true;
                }
                success_count.fetch_add(1);
            }

            // Log slow processing
            auto task_end = std::chrono::high_resolution_clock::now();
            double task_ms = std::chrono::duration<double, std::milli>(task_end - task_start).count();
            if (task_ms > 1000) {
                log(WARNING, "Slow tensor loading for tensor " + std::to_string(i) +
                           ": " + std::to_string(task_ms) + "ms");
            }

            return success;
        }));

        // Log progress periodically
        if ((i+1) % 100 == 0 || (i+1) == tensor_count) {
            log(INFO, "Queued tensor loading: " + std::to_string(i+1) + "/" +
                     std::to_string(tensor_count) + " tensors");
        }
    }

    // 14. Wait for all loading tasks to complete
    int progress_counter = 0;
    for (auto& future : load_futures) {
        future.get();
        progress_counter++;

        if (progress_counter % 100 == 0 || progress_counter == load_futures.size()) {
            log(INFO, "Tensor loading progress: " + std::to_string(progress_counter) + "/" +
                     std::to_string(load_futures.size()) + " tensors");
        }
    }

    // 15. Update final metadata
    file_metadata->preloaded_count = success_count.load();
    file_metadata->tensor_data_size = current_offset.load();

    // 16. Update global stats
    g_stats.preloaded_count += success_count.load();
    g_stats.total_memory += current_offset.load();

    // 17. Clean up resources
    SharedMemory::detach_segment(shmaddr);
    SharedMemory::detach_segment(tensor_data);

    // 18. Report results
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    log(INFO, "Successfully preloaded " + std::to_string(success_count.load()) +
             " of " + std::to_string(tensor_count) + " tensors from " + filepath +
             " (" + std::to_string(current_offset.load() / (1024 * 1024)) + "MB) in " +
             std::to_string(elapsed_seconds) + " seconds (" +
             std::to_string(success_count.load() / elapsed_seconds) + " tensors/sec)");

    return success_count.load() > 0;
}

// Global preloader function for multiple files
bool preload_tensors(py::list file_tensors, int num_threads) {
    try {
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
            if (!preload_file_tensors(filepath, tensor_infos, file_threads)) {
                log(WARNING, "Failed to preload tensors for " + filepath);
                // Continue with next file
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
    stats["read_time"] = g_stats.read_time_ns.load() / 1e9; // Convert to seconds
    stats["deserialize_time"] = g_stats.deserialize_time_ns.load() / 1e9;
    stats["copy_time"] = g_stats.copy_time_ns.load() / 1e9;
    return stats;
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
    m.def("cleanup_shared_memory", &SharedMemory::cleanup_all_segments,
          "Clean up all shared memory segments created by this module");
}