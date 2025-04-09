// Updated memory_mapped_file.cpp

#include "memory_mapped_file.h"
#include "gloo_file_broadcast.h"

MemoryMappedFile::MemoryMappedFile(const std::string& path, GlooFileBroadcast* broadcaster)
    : fd(-1), mapped_data(nullptr), file_size(0), is_valid(false),
      filepath(path), broadcast_bandwidth_gbps(0.0), is_broadcast_mode(false) {

    // Empty path means this is a base constructor call for a derived class
    if (path.empty()) {
        return;
    }

    // If we have a broadcaster and it has multiple ranks, use broadcast mode
    if (broadcaster && broadcaster->getWorldSize() > 1) {
        is_broadcast_mode = true;

        // Use the broadcaster to get the file
        size_t receivedSize = 0;
        mapped_data = broadcaster->broadcastFile(path, receivedSize);

        if (mapped_data == nullptr) {
            log(ERROR, "Failed to broadcast file: " + path);
            return;
        }

        // Store the file size
        file_size = receivedSize;

        // Store bandwidth for reporting
        broadcast_bandwidth_gbps = broadcaster->getLastBroadcastBandwidthGBs();

        log(INFO, "Successfully received broadcast file: " + path +
                 " (" + std::to_string(file_size / (1024 * 1024)) + " MB, " +
                 std::to_string(broadcast_bandwidth_gbps) + " GB/s)");
    }
    else {
        // Standard local file loading
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

        log(INFO, "Successfully memory-mapped file: " + path + " (" +
                 std::to_string(file_size / (1024 * 1024)) + " MB)");
    }

    is_valid = true;
}

MemoryMappedFile::~MemoryMappedFile() {
    if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
        // Only unmap if it's a real file mapping
        if (is_broadcast_mode) {
            // For broadcast mode, we need to munmap regardless of fd
            munmap(mapped_data, file_size);
        } else if (fd != -1) {
            // For local files, we only unmap if the fd is valid
            munmap(mapped_data, file_size);
        }
    }
    if (fd != -1) {
        close(fd);
    }
}

bool MemoryMappedFile::valid() const {
    return is_valid;
}

const char* MemoryMappedFile::data() const {
    return static_cast<const char*>(mapped_data);
}

size_t MemoryMappedFile::size() const {
    return file_size;
}

const char* MemoryMappedFile::get_tensor_data(size_t offset, size_t length) const {
    if (!is_valid || offset + length > file_size) {
        log(ERROR, "Invalid tensor access in memory-mapped file: " + filepath);
        return nullptr;
    }
    return static_cast<const char*>(mapped_data) + offset;
}

bool MemoryMappedFile::isBroadcastMode() const {
    return is_broadcast_mode;
}

double MemoryMappedFile::getBroadcastBandwidthGBps() const {
    return broadcast_bandwidth_gbps;
}