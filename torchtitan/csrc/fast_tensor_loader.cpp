// fast_tensor_loader.cpp with parallel processing
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <mutex>
#include <chrono>

// Mutex for thread-safe console output
std::mutex cout_mutex;

// Function for logging with thread safety
void safe_log(const std::string& message) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << message << std::endl;
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
        safe_log(std::string("Error in tensor deserialization: ") + e.what());
        return torch::Tensor(); // Return empty tensor
    } catch (const std::exception& e) {
        safe_log(std::string("Exception in fast_load_tensor_from_memory: ") + e.what());
        return torch::Tensor(); // Return empty tensor
    }
}

// Function that copies file content into a buffer first
std::string read_file_to_string(py::object fileobj) {
    py::gil_scoped_acquire acquire;  // Ensure GIL is held for Python operations
    try {
        // Explicitly seek to the beginning to ensure we read the whole file
        if (py::hasattr(fileobj, "seek")) {
            fileobj.attr("seek")(0);
        }

        // For file objects, read all data at once
        py::object data_obj = fileobj.attr("read")(-1);
        return data_obj.cast<std::string>();
    } catch (const std::exception& e) {
        safe_log(std::string("Error reading file: ") + e.what());
        return "";
    }
}

// Improved function for loading from file object
torch::Tensor fast_load_tensor_from_fileobj(py::object fileobj,
                                           const std::vector<int64_t>& offsets,
                                           const std::vector<int64_t>& lengths) {
    // Get the file content as a string buffer
    std::string data = read_file_to_string(fileobj);
    if (data.empty()) {
        safe_log("Empty data read from file");
        return torch::Tensor();
    }

    // Use the memory-based loader
    return fast_load_tensor_from_memory(data, offsets, lengths);
}

// Parallel processing function
std::vector<torch::Tensor> fast_load_tensors_parallel(
    const std::vector<py::object>& fileobjs,
    const std::vector<std::vector<int64_t>>& offsets_list,
    const std::vector<std::vector<int64_t>>& lengths_list,
    int num_threads = -1) {

    // Set number of threads if not specified
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    } else {
        num_threads = std::min(num_threads, omp_get_max_threads());
    }

    // Prepare output vector
    size_t batch_size = fileobjs.size();
    std::vector<torch::Tensor> results(batch_size);

    safe_log("Processing " + std::to_string(batch_size) + " tensors with " +
             std::to_string(num_threads) + " threads");

    // First, read all files into memory (sequentially)
    std::vector<std::string> file_contents(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        py::gil_scoped_acquire gil;  // Acquire GIL for Python operations
        file_contents[i] = read_file_to_string(fileobjs[i]);
    }

    // Set OpenMP thread count
    omp_set_num_threads(num_threads);

    // Process tensors in parallel from memory buffers
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(batch_size); i++) {
        // Skip if empty data
        if (file_contents[i].empty()) {
            continue;
        }

        int thread_id = omp_get_thread_num();
        auto start_time = std::chrono::high_resolution_clock::now();

        // Get parameters for this request
        const std::vector<int64_t>& offsets = (i < offsets_list.size()) ? offsets_list[i] : std::vector<int64_t>();
        const std::vector<int64_t>& lengths = (i < lengths_list.size()) ? lengths_list[i] : std::vector<int64_t>();

        try {
            // Process the tensor from the memory buffer
            torch::Tensor tensor = fast_load_tensor_from_memory(file_contents[i], offsets, lengths);
            results[i] = tensor;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Thread " << thread_id << " processed tensor " << i
                      << " in " << duration << "ms" << std::endl;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Error processing tensor " << i << ": " << e.what() << std::endl;
        }
    }

    return results;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_load_tensor_from_memory", &fast_load_tensor_from_memory,
          "Fast tensor loading from memory buffer",
          py::arg("data"),
          py::arg("offsets") = std::vector<int64_t>(),
          py::arg("lengths") = std::vector<int64_t>());

    m.def("fast_load_tensor_from_fileobj", &fast_load_tensor_from_fileobj,
          "Fast tensor loading directly from file object",
          py::arg("fileobj"),
          py::arg("offsets") = std::vector<int64_t>(),
          py::arg("lengths") = std::vector<int64_t>());

    m.def("fast_load_tensors_parallel", &fast_load_tensors_parallel,
          "Load multiple tensors in parallel",
          py::arg("fileobjs"),
          py::arg("offsets_list") = std::vector<std::vector<int64_t>>(),
          py::arg("lengths_list") = std::vector<std::vector<int64_t>>(),
          py::arg("num_threads") = -1);
}