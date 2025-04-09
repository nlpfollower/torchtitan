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
class GlooFileBroadcast {
public:
    // Constructor
    GlooFileBroadcast(int rank, int worldSize,
                        const std::string& redisHost, int redisPort,
                        const std::string& runId);

    // Destructor
    ~GlooFileBroadcast();

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
    int rank;
    int worldSize;
    std::string redisHost;
    int redisPort;
    std::string runId;
    std::string prefix;
    bool initialized;

    // Store and contexts
    std::shared_ptr<gloo::rendezvous::Store> store;
    std::vector<std::shared_ptr<gloo::rendezvous::Context>> contexts;

    // Network interfaces
    std::vector<std::pair<std::string, std::string>> usableInterfaces;

    // Performance metrics
    std::chrono::milliseconds lastBroadcastDuration;
    double lastBroadcastBandwidthGBs;

    // Default optimal chunk size 1GB
    size_t maxChunkSizeBytes = 1ULL * 1024 * 1024 * 1024;

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