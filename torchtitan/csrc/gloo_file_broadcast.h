#pragma once

#include <gloo/broadcast.h>
#include <gloo/transport/tcp/device.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/redis_store.h>
#include <gloo/rendezvous/context.h>

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

// Simple broadcaster class that memory maps a file on rank 0 and distributes it to all other ranks
class GlooFileBroadcaster {
public:
    // Constructor
    GlooFileBroadcaster(int rank, int worldSize,
                        const std::string& redisHost, int redisPort,
                        const std::string& runId);

    // Destructor
    ~GlooFileBroadcaster();

    // Initialize the broadcaster (discover interfaces, setup contexts)
    bool initialize();

    // Cleanup and release resources
    void destroy();

    // Broadcast a file from rank 0 to all other ranks
    // Returns a pointer to the mapped data on success, nullptr on failure
    // Also returns the size of the file via outDataSize
    void* broadcastFile(const std::string& filePath, size_t& outDataSize);

    int getWorldSize() const;

    // Get bandwidth of last broadcast operation in GB/s
    double getLastBroadcastBandwidthGBs() const;

private:
    // Configuration
    int rank_;
    int worldSize_;
    std::string redisHost_;
    int redisPort_;
    std::string runId_;
    std::string prefix_;
    bool initialized_;

    // Store and contexts
    std::shared_ptr<gloo::rendezvous::Store> store_;
    std::vector<std::shared_ptr<gloo::rendezvous::Context>> contexts_;

    // Network interfaces
    std::vector<std::pair<std::string, std::string>> usableInterfaces_;

    // Performance metrics
    std::chrono::milliseconds lastBroadcastDuration_;
    double lastBroadcastBandwidthGBs_;

    // Default optimal chunk size 1GB
    size_t maxChunkSizeBytes_ = 1ULL * 1024 * 1024 * 1024;

    // Helper methods
    void log(const std::string& message);
    std::vector<std::pair<std::string, std::string>> discoverNetworkInterfaces();
    bool coordinateInterfaces();
    bool setupContexts();
    bool executeBarrier(const std::string& barrierName);
    
    // Helpers for store communication
    std::vector<char> stringToVector(const std::string& str);
    std::string vectorToString(const std::vector<char>& vec);
};