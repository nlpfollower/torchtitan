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
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>
#include <set>
#include <condition_variable>
#include <queue>
#include <memory>

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
    size_t shm_offset;   // Offset within shared memory segment
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

// Deserialize a tensor from memory
torch::Tensor load_tensor_from_memory(const char* data, size_t length) {
    try {
        if (data == nullptr || length == 0) {
            log(ERROR, "Invalid data or length in load_tensor_from_memory");
            return torch::Tensor();
        }

        // Create a copy with pre-allocated memory - use vector as required by pickle_load
        std::vector<char> data_vec(data, data + length);

        // Use the vector with pickle_load
        try {
            auto ivalue = torch::pickle_load(data_vec);
            if (!ivalue.isTensor()) {
                log(ERROR, "Deserialized data is not a tensor");
                return torch::Tensor();
            }

            torch::Tensor tensor = ivalue.toTensor().cpu();
            if (!tensor.defined()) {
                log(ERROR, "Failed to create tensor from deserialized data");
                return torch::Tensor();
            }

            return tensor;
        } catch (const c10::Error& e) {
            log(ERROR, "PyTorch error during deserialization: " + std::string(e.what()));
            return torch::Tensor();
        }
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
    std::atomic<uint64_t> read_time_ns{0};       // Store nanoseconds as integers
    std::atomic<uint64_t> deserialize_time_ns{0}; // Store nanoseconds as integers
    std::atomic<uint64_t> copy_time_ns{0};       // Store nanoseconds as integers
} g_stats;

// Helper function to atomically add seconds to a nanosecond counter
inline void atomic_add_seconds(std::atomic<uint64_t>& counter, double seconds) {
    uint64_t ns = static_cast<uint64_t>(seconds * 1e9);
    counter.fetch_add(ns, std::memory_order_relaxed);
}
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

// ThreadPool implementation for parallel processing
class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });

                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// Memory-mapped file class
class MemoryMappedFile {
private:
    int fd;
    void* mapped_data;
    size_t file_size;
    bool is_valid;
    std::string filepath;

public:
    MemoryMappedFile(const std::string& path) : fd(-1), mapped_data(nullptr), file_size(0), is_valid(false), filepath(path) {
        // Open the file
        fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) {
            log(ERROR, "Failed to open file for memory mapping: " + path);
            return;
        }

        // Get file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            log(ERROR, "Failed to get file size for memory mapping: " + path);
            close(fd);
            return;
        }
        file_size = sb.st_size;

        // Map the file into memory
        mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) {
            log(ERROR, "Failed to memory map file: " + path);
            close(fd);
            mapped_data = nullptr;
            return;
        }

        is_valid = true;
        log(INFO, "Successfully memory-mapped file: " + path + " (" +
                 std::to_string(file_size / (1024 * 1024)) + " MB)");
    }

    ~MemoryMappedFile() {
        if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
            munmap(mapped_data, file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    bool valid() const { return is_valid; }

    const char* data() const {
        return static_cast<const char*>(mapped_data);
    }

    size_t size() const { return file_size; }

    const char* get_tensor_data(size_t offset, size_t length) const {
        if (!is_valid || offset + length > file_size) {
            log(ERROR, "Invalid tensor access in memory-mapped file: " + filepath);
            return nullptr;
        }
        return static_cast<const char*>(mapped_data) + offset;
    }
};

// Structure to hold tensor information for parallel processing
struct TensorTask {
    int tensor_idx;
    int64_t offset;
    int64_t length;
    std::string key;
    std::string filepath;
};

// Optimized version of preload_file_tensors for large files
bool preload_file_tensors_optimized(
    const std::string& filepath,
    const std::vector<py::dict>& tensor_infos,
    int num_threads
) {
    try {
        auto start_time = std::chrono::high_resolution_clock::now();

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

        // Memory-map the file for efficient read access
        int fd = open(filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            log(ERROR, "Failed to open file: " + filepath);
            return false;
        }

        // Get file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            log(ERROR, "Failed to get file size: " + filepath);
            close(fd);
            return false;
        }
        size_t file_size = sb.st_size;

        // Map the file into memory
        auto mmap_start_time = std::chrono::high_resolution_clock::now();
        void* mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) {
            log(ERROR, "Failed to memory map file: " + filepath);
            close(fd);
            return false;
        }

        // Cast to char* for byte-level access
        const char* file_data = static_cast<const char*>(mapped_data);

        log(INFO, "Successfully memory-mapped file: " + filepath + " (" +
               std::to_string(file_size / (1024 * 1024)) + " MB)");
        auto mmap_end_time = std::chrono::high_resolution_clock::now();
        double mmap_elapsed_seconds = std::chrono::duration<double>(mmap_end_time - mmap_start_time).count();

        log(INFO, "Mmap took " + std::to_string(mmap_elapsed_seconds) + " seconds");

        // Use a ThreadPool to parallelize tensor size estimation
        int worker_threads = (num_threads > 0) ? num_threads : std::thread::hardware_concurrency();

        // Create a thread pool
        std::vector<std::thread> workers;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<bool> stop(false);
        std::queue<std::function<void()>> tasks;
        std::atomic<int> active_workers(0);

        // Create worker threads
        for (int i = 0; i < worker_threads; i++) {
            workers.emplace_back([&]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [&] {
                            return stop.load() || !tasks.empty();
                        });

                        if (stop.load() && tasks.empty()) {
                            return;
                        }

                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    active_workers++;
                    task();
                    active_workers--;
                    condition.notify_one(); // Notify for task completion
                }
            });
        }

        // Helper to add a task to the queue
        auto enqueue_task = [&](std::function<void()> task) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                tasks.push(std::move(task));
            }
            condition.notify_one();
        };

        log(INFO, "Using " + std::to_string(worker_threads) + " worker threads for tensor preloading");

        // Prepare task information
        std::vector<int64_t> tensor_offsets(tensor_count);
        std::vector<int64_t> tensor_lengths(tensor_count);
        std::vector<std::string> tensor_keys(tensor_count);

        for (int i = 0; i < tensor_count; i++) {
            py::dict info = tensor_infos[i];
            tensor_offsets[i] = py::cast<int64_t>(info["offset"]);
            tensor_lengths[i] = py::cast<int64_t>(info["length"]);
            tensor_keys[i] = py::cast<std::string>(info["key"]);
        }

        // Create shared memory for tensor metadata
        size_t metadata_size = sizeof(TensorMetadata) * tensor_count + sizeof(FileMetadata);
        int tensor_meta_shmid = shmget(tensor_meta_key, metadata_size, IPC_CREAT | 0666);
        if (tensor_meta_shmid == -1) {
            log(ERROR, "Failed to create tensor metadata shared memory");
            munmap(mapped_data, file_size);
            close(fd);
            return false;
        }
        register_shm_segment(tensor_meta_shmid);

        // Attach to tensor metadata segment
        void* shmaddr = shmat(tensor_meta_shmid, NULL, 0);
        if (shmaddr == (void*)-1) {
            log(ERROR, "Failed to attach to tensor metadata shared memory");
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            munmap(mapped_data, file_size);
            close(fd);
            return false;
        }

        // The layout is: FileMetadata followed by array of TensorMetadata
        FileMetadata* file_metadata = static_cast<FileMetadata*>(shmaddr);
        TensorMetadata* tensor_metadata = reinterpret_cast<TensorMetadata*>(file_metadata + 1);

        // Initialize file metadata
        file_metadata->tensor_count = tensor_count;
        file_metadata->preloaded_count = 0;
        file_metadata->tensor_data_size = 0;

        // Initialize tensor metadata
        for (int i = 0; i < tensor_count; i++) {
            tensor_metadata[i].offset = tensor_offsets[i];
            tensor_metadata[i].length = tensor_lengths[i];
            tensor_metadata[i].is_preloaded = false;
            strncpy(tensor_metadata[i].key, tensor_keys[i].c_str(), sizeof(tensor_metadata[i].key) - 1);
            tensor_metadata[i].key[sizeof(tensor_metadata[i].key) - 1] = '\0';
        }

        // Synchronization and shared variables
        std::mutex tensor_sizes_mutex;
        std::atomic<size_t> total_tensor_size(0);
        std::vector<size_t> tensor_sizes(tensor_count, 0);
        std::atomic<int> processed_count(0);

        // First pass: Determine tensor sizes in parallel
        for (int i = 0; i < tensor_count; i++) {
            enqueue_task([&, i]() {
                try {
                    auto task_start = std::chrono::high_resolution_clock::now();

                    // Get tensor data from memory-mapped file
                    auto read_start = std::chrono::high_resolution_clock::now();
                    const char* tensor_data = file_data + tensor_offsets[i];
                    auto read_end = std::chrono::high_resolution_clock::now();
                    double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
                    atomic_add_seconds(g_stats.read_time_ns, read_ms / 1000.0);

                    if (tensor_data == nullptr || tensor_offsets[i] + tensor_lengths[i] > file_size) {
                        log(ERROR, "Invalid tensor offset or length for tensor " + std::to_string(i));
                        return;
                    }

                    // Deserialize the tensor to get its size
                    auto deserialize_start = std::chrono::high_resolution_clock::now();

                    // Create a vector copy as required by pickle_load
                    std::vector<char> data_vec(tensor_data, tensor_data + tensor_lengths[i]);
                    torch::Tensor tensor;

                    try {
                        auto ivalue = torch::pickle_load(data_vec);
                        if (ivalue.isTensor()) {
                            tensor = ivalue.toTensor().cpu();
                        }
                    } catch (const c10::Error& e) {
                        log(ERROR, "PyTorch error deserializing tensor " + std::to_string(i) +
                                 ": " + e.what());
                    }

                    auto deserialize_end = std::chrono::high_resolution_clock::now();
                    double deserialize_ms = std::chrono::duration<double, std::milli>(deserialize_end - deserialize_start).count();
                    atomic_add_seconds(g_stats.deserialize_time_ns, deserialize_ms / 1000.0);

                    if (tensor.defined()) {
                        if (!tensor.is_contiguous()) {
                            tensor = tensor.contiguous();
                        }

                        // Update metadata (thread-safe)
                        {
                            std::lock_guard<std::mutex> lock(tensor_sizes_mutex);
                            tensor_metadata[i].dtype = static_cast<int64_t>(tensor.scalar_type());
                            tensor_metadata[i].ndim = tensor.dim();
                            for (int64_t d = 0; d < tensor.dim() && d < 8; d++) {
                                tensor_metadata[i].dims[d] = tensor.size(d);
                            }

                            // Store size
                            tensor_sizes[i] = tensor.nbytes();
                        }

                        // Update total size (thread-safe with atomic)
                        total_tensor_size.fetch_add(tensor_sizes[i]);
                    } else {
                        log(WARNING, "Failed to deserialize tensor " + std::to_string(i));
                    }

                    // Update processed count
                    int current = processed_count.fetch_add(1) + 1;
                    if (current % 100 == 0 || current == tensor_count) {
                        log(INFO, "Size calculation progress: " + std::to_string(current) + "/" +
                                 std::to_string(tensor_count) + " tensors");
                    }

                    auto task_end = std::chrono::high_resolution_clock::now();
                    double task_ms = std::chrono::duration<double, std::milli>(task_end - task_start).count();
                    if (task_ms > 1000) { // Log slow tensor processing (>1s)
                        log(WARNING, "Slow tensor size calculation for tensor " + std::to_string(i) +
                                   ": " + std::to_string(task_ms) + "ms");
                    }
                } catch (const std::exception& e) {
                    log(ERROR, "Exception processing tensor " + std::to_string(i) +
                             " in size calculation: " + e.what());
                }
            });
        }

        // Wait for all tasks to complete
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [&]() {
                return processed_count.load() == tensor_count;
            });
        }

        if (total_tensor_size.load() == 0) {
            log(ERROR, "No valid tensors found in " + filepath);
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            munmap(mapped_data, file_size);
            close(fd);
            return false;
        }

        // Add a reasonable buffer - 2% instead of 5%
        size_t final_size = total_tensor_size.load() + (total_tensor_size.load() / 50);

        log(INFO, "Size calculation complete. Total tensor data size: " +
                 std::to_string(final_size / (1024 * 1024)) + "MB");

        // Create shared memory for tensor data
        log(INFO, "Creating shared memory segment of " +
                 std::to_string(final_size / (1024 * 1024)) + "MB for tensor data");

        // Use advisory huge pages for large allocations if available
        const size_t HUGE_PAGE_THRESHOLD = 1024 * 1024 * 1024; // 1GB
        int tensor_data_shmid;

        if (final_size >= HUGE_PAGE_THRESHOLD) {
            // Try with huge pages first
            tensor_data_shmid = shmget(tensor_data_key, final_size, IPC_CREAT | 0666 | SHM_HUGETLB);
            if (tensor_data_shmid == -1) {
                log(WARNING, "Failed to create huge page shared memory, falling back to standard pages");
                tensor_data_shmid = shmget(tensor_data_key, final_size, IPC_CREAT | 0666);
            } else {
                log(INFO, "Successfully allocated shared memory using huge pages");
            }
        } else {
            tensor_data_shmid = shmget(tensor_data_key, final_size, IPC_CREAT | 0666);
        }

        if (tensor_data_shmid == -1) {
            log(ERROR, "Failed to create tensor data shared memory: " +
                     std::to_string(errno));
            shmdt(shmaddr);
            shmctl(tensor_meta_shmid, IPC_RMID, NULL);
            munmap(mapped_data, file_size);
            close(fd);
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
            munmap(mapped_data, file_size);
            close(fd);
            return false;
        }

        // Reset counters for second pass
        processed_count.store(0);
        std::atomic<size_t> current_offset(0);
        std::atomic<int> success_count(0);

        // Second pass: Copy tensors into shared memory in parallel
        for (int i = 0; i < tensor_count; i++) {
            // Skip if we already know it's invalid
            if (tensor_sizes[i] == 0) {
                processed_count.fetch_add(1);
                continue;
            }

            enqueue_task([&, i]() {
                try {
                    auto task_start = std::chrono::high_resolution_clock::now();

                    // Get tensor data from memory-mapped file
                    auto read_start = std::chrono::high_resolution_clock::now();
                    const char* tensor_data_ptr = file_data + tensor_offsets[i];
                    auto read_end = std::chrono::high_resolution_clock::now();
                    double read_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
                    atomic_add_seconds(g_stats.read_time_ns, read_ms / 1000.0);

                    // Deserialize the tensor
                    auto deserialize_start = std::chrono::high_resolution_clock::now();

                    // Create a vector copy as required by pickle_load
                    std::vector<char> data_vec(tensor_data_ptr, tensor_data_ptr + tensor_lengths[i]);
                    torch::Tensor tensor;

                    try {
                        auto ivalue = torch::pickle_load(data_vec);
                        if (ivalue.isTensor()) {
                            tensor = ivalue.toTensor().cpu();
                        }
                    } catch (const c10::Error& e) {
                        log(ERROR, "PyTorch error deserializing tensor " + std::to_string(i) +
                                 ": " + e.what());
                    }

                    auto deserialize_end = std::chrono::high_resolution_clock::now();
                    double deserialize_ms = std::chrono::duration<double, std::milli>(deserialize_end - deserialize_start).count();
                    atomic_add_seconds(g_stats.deserialize_time_ns, deserialize_ms / 1000.0);

                    if (tensor.defined()) {
                        if (!tensor.is_contiguous()) {
                            tensor = tensor.contiguous();
                        }

                        // Reserve space in shared memory (atomic operation)
                        size_t tensor_size = tensor_sizes[i];
                        size_t offset = current_offset.fetch_add(tensor_size);

                        // Check if we have enough space
                        if (offset + tensor_size <= final_size) {
                            // Copy to shared memory
                            auto copy_start = std::chrono::high_resolution_clock::now();

                            // Use direct memcpy for speed
                            memcpy(static_cast<char*>(tensor_data) + offset,
                                   tensor.data_ptr(),
                                   tensor_size);

                            auto copy_end = std::chrono::high_resolution_clock::now();
                            double copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
                            atomic_add_seconds(g_stats.copy_time_ns, copy_ms / 1000.0);

                            // Update metadata atomically
                            {
                                std::lock_guard<std::mutex> lock(tensor_sizes_mutex);
                                tensor_metadata[i].shm_offset = offset;
                                tensor_metadata[i].is_preloaded = true;
                            }

                            success_count.fetch_add(1);
                        } else {
                            log(ERROR, "Not enough shared memory for tensor " + std::to_string(i));
                        }
                    }

                    // Update processed count
                    int current = processed_count.fetch_add(1) + 1;
                    if (current % 100 == 0 || current == tensor_count) {
                        log(INFO, "Tensor loading progress: " + std::to_string(current) + "/" +
                                 std::to_string(tensor_count) + " tensors");
                    }

                    auto task_end = std::chrono::high_resolution_clock::now();
                    double task_ms = std::chrono::duration<double, std::milli>(task_end - task_start).count();
                    if (task_ms > 1000) { // Log slow tensor processing (>1s)
                        log(WARNING, "Slow tensor loading for tensor " + std::to_string(i) +
                                   ": " + std::to_string(task_ms) + "ms");
                    }
                } catch (const std::exception& e) {
                    log(ERROR, "Exception processing tensor " + std::to_string(i) +
                             " in tensor loading: " + e.what());
                    processed_count.fetch_add(1);
                }
            });
        }

        // Wait for all tasks to complete
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            condition.wait(lock, [&]() {
                return processed_count.load() == tensor_count;
            });
        }

        // Stop the thread pool
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop.store(true);
            condition.notify_all();
        }

        // Join all threads
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        // Update final metadata
        file_metadata->preloaded_count = success_count.load();
        file_metadata->tensor_data_size = current_offset.load();

        // Update global stats
        g_stats.preloaded_count += success_count.load();
        g_stats.total_memory += current_offset.load();

        // Clean up
        shmdt(shmaddr);
        shmdt(tensor_data);
        munmap(mapped_data, file_size);
        close(fd);

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

        log(INFO, "Successfully preloaded " + std::to_string(success_count.load()) +
                 " of " + std::to_string(tensor_count) + " tensors from " + filepath +
                 " (" + std::to_string(current_offset.load() / (1024 * 1024)) + "MB) in " +
                 std::to_string(elapsed_seconds) + " seconds (" +
                 std::to_string(success_count.load() / elapsed_seconds) + " tensors/sec)");

        return success_count.load() > 0;
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

// Main function to preload tensors
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

            // Preload tensors for this file using the optimized function
            if (!preload_file_tensors_optimized(filepath, tensor_infos, num_threads)) {
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
        log(INFO, "  Read time: " + std::to_string(g_stats.read_time_ns) + " seconds");
        log(INFO, "  Deserialize time: " + std::to_string(g_stats.deserialize_time_ns) + " seconds");
        log(INFO, "  Copy time: " + std::to_string(g_stats.copy_time_ns) + " seconds");

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
    stats["read_time"] = g_stats.read_time_ns.load();
    stats["deserialize_time"] = g_stats.deserialize_time_ns.load();
    stats["copy_time"] = g_stats.copy_time_ns.load();
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