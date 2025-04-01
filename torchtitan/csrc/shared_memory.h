#pragma once

#include <set>
#include <mutex>
#include <string>
#include <sys/shm.h>
#include "tensor_common.h"

class SharedMemory {
public:
    // Register segments for cleanup
    static void register_segment(int shmid);
    
    // Clean up specific file's shared memory segments
    static bool cleanup_file_segments(const std::string& filepath);
    
    // Clean up all registered segments
    static bool cleanup_all_segments();
    
    // Create a shared memory segment
    static int create_segment(int key, size_t size, bool use_huge_pages = false);
    
    // Attach to a shared memory segment
    static void* attach_segment(int shmid);
    
    // Detach from a shared memory segment
    static bool detach_segment(void* shmaddr);
    
    // Mark a segment for deletion
    static bool mark_for_deletion(int shmid);

private:
    static std::set<int> created_segments;
    static std::mutex segments_mutex;
};