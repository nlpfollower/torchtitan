// Updated memory_mapped_file.h

#pragma once

#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "tensor_common.h"

// Forward declaration
class GlooFileBroadcast;

class MemoryMappedFile {
public:
    // Unified constructor - handles both local and broadcast modes
    MemoryMappedFile(const std::string& path, GlooFileBroadcast* broadcaster = nullptr);

    // Virtual destructor for inheritance
    virtual ~MemoryMappedFile();

    // Interface methods
    virtual bool valid() const;
    virtual const char* data() const;
    virtual size_t size() const;
    virtual const char* get_tensor_data(size_t offset, size_t length) const;

    // Check if file was loaded via broadcast
    bool isBroadcastMode() const;

    // Get broadcast performance (0 if not broadcast mode)
    double getBroadcastBandwidthGBps() const;

protected:
    // Protected members
    int fd;
    void* mapped_data;
    size_t file_size;
    bool is_valid;
    std::string filepath;
    double broadcast_bandwidth_gbps;
    bool is_broadcast_mode;
};