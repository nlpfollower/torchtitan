// model_mapper.cpp
// Responsible for memory-mapping model files into shared memory for efficient access

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <thread>
#include <csignal>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstring>

// Global constants
#define SHM_META_KEY_BASE 9000  // Base key for metadata segments
#define SHM_DATA_KEY_BASE 10000 // Base key for data segments

// Structure to store file metadata in shared memory
struct FileMetadata {
    size_t file_size;
    uint32_t checksum;
    char filepath[256];  // Fixed-size buffer for filepath
};

// Global state to track mapped files
struct MappedFile {
    std::string filepath;
    int meta_shmid;      // Shared memory ID for metadata
    int data_shmid;      // Shared memory ID for file data
    FileMetadata* metadata;
    void* data;
    size_t file_size;
    int fd;
};

std::map<std::string, MappedFile> mapped_files;
bool running = true;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    running = false;
}

// Calculate checksum for verification
uint32_t calculate_checksum(const unsigned char* data, size_t size) {
    uint32_t checksum = 0;
    for (size_t i = 0; i < size; ++i) {
        checksum += data[i];
    }
    return checksum;
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

// Map a model file into shared memory
bool map_file(const std::string& filepath) {
    // Check if already mapped
    if (mapped_files.find(filepath) != mapped_files.end()) {
        std::cout << "File already mapped: " << filepath << std::endl;
        return true;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Open the file
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) == -1) {
        std::cerr << "Failed to get file size: " << filepath << std::endl;
        close(fd);
        return false;
    }

    // Read file into buffer
    unsigned char* file_buffer = new unsigned char[st.st_size];
    ssize_t bytes_read = read(fd, file_buffer, st.st_size);
    if (bytes_read != st.st_size) {
        std::cerr << "Failed to read file: " << filepath << std::endl;
        delete[] file_buffer;
        close(fd);
        return false;
    }

    // Calculate checksum for verification
    uint32_t checksum = calculate_checksum(file_buffer, st.st_size);

    // Generate keys for this file
    int meta_key = generate_file_key(filepath, SHM_META_KEY_BASE);
    int data_key = generate_file_key(filepath, SHM_DATA_KEY_BASE);

    // Create shared memory segment for metadata
    int meta_shmid = shmget(meta_key, sizeof(FileMetadata), IPC_CREAT | 0666);
    if (meta_shmid == -1) {
        std::cerr << "Failed to create metadata shared memory for " << filepath << std::endl;
        delete[] file_buffer;
        close(fd);
        return false;
    }

    // Create shared memory segment for file data
    int data_shmid = shmget(data_key, st.st_size, IPC_CREAT | 0666);
    if (data_shmid == -1) {
        std::cerr << "Failed to create data shared memory for " << filepath << std::endl;
        shmctl(meta_shmid, IPC_RMID, NULL);
        delete[] file_buffer;
        close(fd);
        return false;
    }

    // Attach to metadata segment
    FileMetadata* metadata = (FileMetadata*)shmat(meta_shmid, NULL, 0);
    if (metadata == (void*)-1) {
        std::cerr << "Failed to attach to metadata shared memory for " << filepath << std::endl;
        shmctl(meta_shmid, IPC_RMID, NULL);
        shmctl(data_shmid, IPC_RMID, NULL);
        delete[] file_buffer;
        close(fd);
        return false;
    }

    // Attach to data segment
    void* data = shmat(data_shmid, NULL, 0);
    if (data == (void*)-1) {
        std::cerr << "Failed to attach to data shared memory for " << filepath << std::endl;
        shmdt(metadata);
        shmctl(meta_shmid, IPC_RMID, NULL);
        shmctl(data_shmid, IPC_RMID, NULL);
        delete[] file_buffer;
        close(fd);
        return false;
    }

    // Populate the metadata
    metadata->file_size = st.st_size;
    metadata->checksum = checksum;
    strncpy(metadata->filepath, filepath.c_str(), sizeof(metadata->filepath) - 1);
    metadata->filepath[sizeof(metadata->filepath) - 1] = '\0';  // Ensure null-termination

    // Copy file data to shared memory
    memcpy(data, file_buffer, st.st_size);

    // Store mapping information
    MappedFile mf;
    mf.filepath = filepath;
    mf.meta_shmid = meta_shmid;
    mf.data_shmid = data_shmid;
    mf.metadata = metadata;
    mf.data = data;
    mf.file_size = st.st_size;
    mf.fd = fd;
    mapped_files[filepath] = mf;

    // Clean up buffer
    delete[] file_buffer;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    std::cout << "Mapped file: " << filepath
              << " (" << (st.st_size / 1024.0 / 1024.0) << " MB)"
              << " in " << duration << "ms"
              << " [meta_key=" << meta_key << ", data_key=" << data_key << "]"
              << std::endl;

    return true;
}

// Unmap a file from shared memory
void unmap_file(const std::string& filepath) {
    auto it = mapped_files.find(filepath);
    if (it != mapped_files.end()) {
        shmdt(it->second.metadata);
        shmdt(it->second.data);
        shmctl(it->second.meta_shmid, IPC_RMID, NULL);
        shmctl(it->second.data_shmid, IPC_RMID, NULL);
        close(it->second.fd);
        mapped_files.erase(it);
        std::cout << "Unmapped file: " << filepath << std::endl;
    }
}

// Clean up all mapped files
void cleanup() {
    for (auto& entry : mapped_files) {
        std::cout << "Unmapping file: " << entry.first << std::endl;
        shmdt(entry.second.metadata);
        shmdt(entry.second.data);
        shmctl(entry.second.meta_shmid, IPC_RMID, NULL);
        shmctl(entry.second.data_shmid, IPC_RMID, NULL);
        close(entry.second.fd);
    }
    mapped_files.clear();
}

int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "Model Mapper (Shared Memory) started (PID: " << getpid() << ")" << std::endl;

    // Parse arguments
    std::string base_directory = "";
    std::vector<std::string> filenames;

    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--checkpoint_path" && i + 1 < argc) {
            base_directory = argv[++i];
            // Ensure directory ends with slash
            if (!base_directory.empty() && base_directory.back() != '/') {
                base_directory += '/';
            }
        } else {
            // Just a filename
            filenames.push_back(arg);
        }
    }

    if (filenames.empty()) {
        // Default to standard model files
        for (int i = 0; i <= 7; i++) {
            filenames.push_back("__0_" + std::to_string(i) + ".distcp");
        }
    }

    // Map all specified files with full paths
    for (const auto& filename : filenames) {
        std::string full_path = base_directory + filename;
        std::cout << "Mapping file: " << full_path << std::endl;
        if (!map_file(full_path)) {
            std::cerr << "Warning: Failed to map " << full_path << std::endl;
        }
    }

    std::cout << "All model files mapped to shared memory. Keeping mappings alive..." << std::endl;
    std::cout << "Press Ctrl+C to exit and unmap files" << std::endl;

    // Keep the program running to maintain mappings
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Clean up before exit
    cleanup();

    std::cout << "Model Mapper shutdown complete" << std::endl;
    return 0;
}