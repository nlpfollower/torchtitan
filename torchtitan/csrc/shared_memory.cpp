#include "shared_memory.h"

// Static member initialization
std::set<int> SharedMemory::created_segments;
std::mutex SharedMemory::segments_mutex;

void SharedMemory::register_segment(int shmid) {
    std::lock_guard<std::mutex> lock(segments_mutex);
    created_segments.insert(shmid);
    log(DEBUG, "Registered shared memory segment " + std::to_string(shmid) + " for cleanup");
}

bool SharedMemory::cleanup_file_segments(const std::string& filepath) {
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
        
        return true;
    } catch (const std::exception& e) {
        log(ERROR, "Error cleaning up shared memory for file " + filepath + ": " + e.what());
        return false;
    }
}

bool SharedMemory::cleanup_all_segments() {
    log(INFO, "Cleaning up shared memory segments");

    // First clean up all registered segments
    {
        std::lock_guard<std::mutex> lock(segments_mutex);
        for (int shmid : created_segments) {
            shmctl(shmid, IPC_RMID, NULL);
        }
        created_segments.clear();
    }

    // Also try to clean up by key ranges
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

int SharedMemory::create_segment(int key, size_t size, bool use_huge_pages) {
    int shmid;
    
    if (use_huge_pages) {
        shmid = shmget(key, size, IPC_CREAT | 0666 | SHM_HUGETLB);
        if (shmid != -1) {
            log(INFO, "Successfully allocated shared memory using huge pages");
        } else {
            log(WARNING, "Failed to create huge page shared memory, falling back to standard pages");
            shmid = shmget(key, size, IPC_CREAT | 0666);
        }
    } else {
        shmid = shmget(key, size, IPC_CREAT | 0666);
    }
    
    if (shmid != -1) {
        register_segment(shmid);
    }
    
    return shmid;
}

void* SharedMemory::attach_segment(int shmid) {
    return shmat(shmid, NULL, 0);
}

bool SharedMemory::detach_segment(void* shmaddr) {
    return shmdt(shmaddr) != -1;
}

bool SharedMemory::mark_for_deletion(int shmid) {
    return shmctl(shmid, IPC_RMID, NULL) != -1;
}