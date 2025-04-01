#pragma once

#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "tensor_common.h"

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string& path);
    ~MemoryMappedFile();
    
    bool valid() const;
    const char* data() const;
    size_t size() const;
    const char* get_tensor_data(size_t offset, size_t length) const;
    
private:
    int fd;
    void* mapped_data;
    size_t file_size;
    bool is_valid;
    std::string filepath;
};