#include "memory_mapped_file.h"

MemoryMappedFile::MemoryMappedFile(const std::string& path)
    : fd(-1), mapped_data(nullptr), file_size(0), is_valid(false), filepath(path) {
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

MemoryMappedFile::~MemoryMappedFile() {
    if (mapped_data != nullptr && mapped_data != MAP_FAILED) {
        munmap(mapped_data, file_size);
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