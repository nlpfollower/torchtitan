// fast_tensor_loader.cpp with native C++ file reading
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h> // For getpid()

// Mutex for thread-safe console output
std::mutex cout_mutex;

// Function for logging with thread safety
void safe_log(const std::string& message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << message << std::endl;
}

// C++ implementation of the Python _ReaderView
class FileView {
private:
    std::string filepath;
    int64_t offset;
    int64_t length;

public:
    FileView(const std::string& path, int64_t off, int64_t len)
        : filepath(path), offset(off), length(len) {}

    // Read the entire content into a string
    std::string read_all() const {
        try {
            // Open the file in binary mode
            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Failed to open file: " + filepath);
            }

            // Seek to the offset
            file.seekg(offset);
            if (!file) {
                throw std::runtime_error("Failed to seek to offset " + std::to_string(offset));
            }

            // Allocate buffer for the data
            std::string buffer(length, '\0');

            // Read the data
            file.read(&buffer[0], length);
            if (file.fail() && !file.eof()) {
                throw std::runtime_error("Failed to read data");
            }

            // Resize the buffer to the actual number of bytes read
            buffer.resize(file.gcount());

            return buffer;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Error reading file: " << e.what() << std::endl;
            return "";
        }
    }
};

// Extract file information from Python fileobj
bool extract_file_info(py::object fileobj, std::string& filepath, int64_t& offset, int64_t& length) {
    py::gil_scoped_acquire gil;  // Acquire GIL for Python operations

    try {
        // Check if it's a _ReaderView
        if (py::hasattr(fileobj, "offset") && py::hasattr(fileobj, "len") && py::hasattr(fileobj, "base_stream")) {
            offset = fileobj.attr("offset").cast<int64_t>();
            length = fileobj.attr("len").cast<int64_t>();

            // Get the base_stream for the file path
            py::object base_stream = fileobj.attr("base_stream");

            // Try to get name attribute (common for file objects)
            if (py::hasattr(base_stream, "name")) {
                filepath = base_stream.attr("name").cast<std::string>();
                return true;
            }
        }

        // Standard file objects
        if (py::hasattr(fileobj, "name")) {
            filepath = fileobj.attr("name").cast<std::string>();

            // Get current position
            int64_t current_pos = 0;
            if (py::hasattr(fileobj, "tell")) {
                current_pos = fileobj.attr("tell")().cast<int64_t>();
            }

            // Reset to beginning
            if (py::hasattr(fileobj, "seek")) {
                fileobj.attr("seek")(0, 0);  // Seek to beginning
            }

            // Get file size
            fileobj.attr("seek")(0, 2);  // Seek to end
            length = fileobj.attr("tell")().cast<int64_t>();  // Get file size

            // Reset position
            fileobj.attr("seek")(current_pos, 0);  // Restore position

            offset = 0;  // Default offset is 0 for regular files
            return true;
        }

        // If we get here, we couldn't extract the file info
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Could not extract file info from Python object" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Error extracting file info: " << e.what() << std::endl;
        return false;
    }
}

// Original function for loading from memory buffer
torch::Tensor fast_load_tensor_from_memory(const std::string& data,
                                           const std::vector<int64_t>& offsets,
                                           const std::vector<int64_t>& lengths) {
    try {
        // Convert string data to vector<char> as required by pickle_load
        std::vector<char> data_vec(data.begin(), data.end());

        // Use pickle_load with the correct data type
        torch::IValue ivalue = torch::pickle_load(data_vec);

        // Try to convert to tensor
        if (!ivalue.isTensor()) {
            safe_log("Loaded data is not a tensor");
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
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Error in tensor deserialization: " << e.what() << std::endl;
        return torch::Tensor(); // Return empty tensor
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Exception in fast_load_tensor_from_memory: " << e.what() << std::endl;
        return torch::Tensor(); // Return empty tensor
    }
}

// Structure to hold file information
struct FileInfo {
    std::string filepath;
    int64_t offset;
    int64_t length;
    size_t index;
    std::vector<int64_t> tensor_offsets;
    std::vector<int64_t> tensor_lengths;
};

// Parallel processing function
std::vector<torch::Tensor> fast_load_tensors_parallel(
    const std::vector<py::object>& fileobjs,
    const std::vector<std::vector<int64_t>>& offsets_list,
    const std::vector<std::vector<int64_t>>& lengths_list,
    int num_threads = -1) {

    // Determine thread count
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    num_threads = std::max(1, num_threads);

    size_t batch_size = fileobjs.size();
    safe_log("Processing " + std::to_string(batch_size) + " tensors with " +
             std::to_string(num_threads) + " threads (PID: " +
             std::to_string(getpid()) + ")");

    // Prepare output vector
    std::vector<torch::Tensor> results(batch_size);

    // Extract file info from all fileobjs (with GIL)
    std::vector<FileInfo> file_infos;
    {
        py::gil_scoped_acquire gil;

        for (size_t i = 0; i < batch_size; ++i) {
            std::string filepath;
            int64_t offset = 0;
            int64_t length = 0;

            if (extract_file_info(fileobjs[i], filepath, offset, length)) {
                FileInfo info;
                info.filepath = filepath;
                info.offset = offset;
                info.length = length;
                info.index = i;
                info.tensor_offsets = (i < offsets_list.size()) ? offsets_list[i] : std::vector<int64_t>();
                info.tensor_lengths = (i < lengths_list.size()) ? lengths_list[i] : std::vector<int64_t>();
                file_infos.push_back(std::move(info));
            } else {
                // Log extraction failure
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "Failed to extract file info for index " << i << std::endl;
            }
        }
    }

    safe_log("Extracted info for " + std::to_string(file_infos.size()) +
             " files out of " + std::to_string(batch_size));

    // No GIL needed beyond this point - process files in parallel

    // Process files in parallel
    std::atomic<size_t> file_index(0);
    std::vector<std::thread> threads;

    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&file_infos, &results, &file_index]() {
            // Get thread ID for logging
            auto thread_id = std::this_thread::get_id();
            std::stringstream ss;
            ss << thread_id;
            std::string thread_id_str = ss.str();

            while (true) {
                // Get next file atomically
                size_t idx = file_index.fetch_add(1);
                if (idx >= file_infos.size()) {
                    break;  // No more files
                }

                const FileInfo& info = file_infos[idx];
                auto start_time = std::chrono::high_resolution_clock::now();

                try {
                    // Create FileView and read data
                    FileView view(info.filepath, info.offset, info.length);
                    std::string data = view.read_all();

                    if (!data.empty()) {
                        // Process the data
                        torch::Tensor tensor = fast_load_tensor_from_memory(
                            data, info.tensor_offsets, info.tensor_lengths
                        );

                        // Store the result
                        results[info.index] = tensor;

                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time
                        ).count();

                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cout << "Thread " << thread_id_str << " processed tensor " << info.index
                                  << " in " << duration << "ms" << std::endl;
                    } else {
                        std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cout << "Thread " << thread_id_str << " got empty data for file "
                                  << info.index << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Thread " << thread_id_str << " error processing file "
                              << info.index << ": " << e.what() << std::endl;
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Handle any fileobjs that we couldn't process with native C++
    std::vector<size_t> unprocessed_indices;
    for (size_t i = 0; i < batch_size; ++i) {
        if (!results[i].defined()) {
            throw std::runtime_error("Failed to process tensor " + std::to_string(i));
        }
    }

    safe_log("All tensors processed");
    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_load_tensor_from_memory", &fast_load_tensor_from_memory,
          "Fast tensor loading from memory buffer",
          py::arg("data"),
          py::arg("offsets") = std::vector<int64_t>(),
          py::arg("lengths") = std::vector<int64_t>());

    m.def("fast_load_tensors_parallel", &fast_load_tensors_parallel,
          "Load multiple tensors in parallel",
          py::arg("fileobjs"),
          py::arg("offsets_list") = std::vector<std::vector<int64_t>>(),
          py::arg("lengths_list") = std::vector<std::vector<int64_t>>(),
          py::arg("num_threads") = -1);
}