#include "gloo_file_broadcast.h"

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Helper function to get current timestamp for logging
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%T") << "."
       << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

// Constructor
GlooFileBroadcast::GlooFileBroadcast(int rank, int worldSize,
                                         const std::string& redisHost, int redisPort,
                                         const std::string& runId)
    : rank(rank),
      worldSize(worldSize),
      redisHost(redisHost),
      redisPort(redisPort),
      runId(runId),
      initialized(false),
      lastBroadcastDuration(0),
      lastBroadcastBandwidthGBs(0.0)
{
    // Set up the prefix for this run
    prefix = "gloo_broadcast_" + runId;
}

// Destructor
GlooFileBroadcast::~GlooFileBroadcast() {
    if (initialized) {
        destroy();
    }
}

// Initialize the broadcaster
bool GlooFileBroadcast::initialize() {
    if (initialized) {
        log("Already initialized");
        return true;
    }

    log("Initializing with Redis host: " + redisHost + ", port: " + std::to_string(redisPort));

    try {
        // Create Redis store for rendezvous
        auto redisStore = std::make_shared<gloo::rendezvous::RedisStore>(redisHost, redisPort);
        store = std::make_shared<gloo::rendezvous::PrefixStore>(prefix, redisStore);

        // If rank 0, clean up any existing keys with this prefix
        if (rank == 0) {
            log("Initializing Redis store");
            store->set("cleanup_complete", stringToVector("done"));
        }

        // Simple barrier using the store to synchronize initialization
        if (!executeBarrier("init")) {
            log("Failed at initial barrier");
            return false;
        }

        // Discover available network interfaces
        usableInterfaces = discoverNetworkInterfaces();
        if (usableInterfaces.empty()) {
            log("No usable network interfaces found");
            return false;
        }

        // Coordinate which interfaces to use
        if (!coordinateInterfaces()) {
            log("Failed to coordinate interfaces");
            return false;
        }

        // Set up contexts for each interface
        if (!setupContexts()) {
            log("Failed to set up contexts");
            return false;
        }

        // Final barrier to ensure all nodes are ready
        if (!executeBarrier("setup_complete")) {
            log("Failed at setup completion barrier");
            return false;
        }

        initialized = true;
        log("Initialization complete with " + std::to_string(contexts.size()) + " active interfaces");
        return true;
    }
    catch (const std::exception& e) {
        log("Error during initialization: " + std::string(e.what()));
        return false;
    }
}

// Clean up resources
void GlooFileBroadcast::destroy() {
    if (!initialized) return;

    log("Cleaning up resources");

    // Clear contexts
    contexts.clear();

    // Clear store
    store.reset();

    initialized = false;
}

// Broadcast a file from rank 0 to all ranks
void* GlooFileBroadcast::broadcastFile(const std::string& filePath, size_t& outDataSize) {
    if (!initialized) {
        log("Cannot broadcast file: not initialized");
        return nullptr;
    }

    void* fileData = nullptr;
    size_t fileSize = 0;

    // Only rank 0 reads the file
    if (rank == 0) {
        log("Opening file for broadcast: " + filePath);

        // Open the file
        int fd = open(filePath.c_str(), O_RDONLY);
        if (fd == -1) {
            log("Failed to open file: " + filePath);
            fileSize = 0;
        } else {
            // Get file size
            struct stat sb;
            if (fstat(fd, &sb) == -1) {
                log("Failed to get file size for: " + filePath);
                close(fd);
                fileSize = 0;
            } else {
                fileSize = sb.st_size;

                // Memory map the file
                fileData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
                if (fileData == MAP_FAILED) {
                    log("Failed to memory map file: " + filePath);
                    close(fd);
                    fileData = nullptr;
                    fileSize = 0;
                } else {
                    log("Successfully memory-mapped file: " + filePath + " (" +
                        std::to_string(fileSize / (1024 * 1024)) + " MB)");

                    // File can be closed after mapping
                    close(fd);
                }
            }
        }
    }

    // Wait at broadcast barrier
    if (!executeBarrier(getUniqueBarrierName("pre_broadcast", filePath))) {
        log("Failed at pre-broadcast barrier");
        if (rank == 0 && fileData != nullptr) {
            munmap(fileData, fileSize);
        }
        return nullptr;
    }

    // Broadcast file size first
    log("Broadcasting file size");

    // All ranks create broadcast opts for file size
    gloo::BroadcastOptions sizeOpts(contexts[0]);
    sizeOpts.setRoot(0);
    sizeOpts.setTag(1);
    sizeOpts.setOutput(&fileSize, sizeof(fileSize));

    // Execute broadcast for file size
    gloo::broadcast(sizeOpts);

    if (fileSize == 0) {
        log("File size is zero or error opening file");
        return nullptr;
    }

    // Non-root ranks allocate memory to receive the file
    if (rank != 0) {
        log("Allocating " + std::to_string(fileSize / (1024 * 1024)) + " MB to receive file");
        fileData = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (fileData == MAP_FAILED) {
            log("Failed to allocate memory for file reception");
            return nullptr;
        }
    }

    // Broadcast the file content
    log("Broadcasting file content");

    auto broadcastStartTime = std::chrono::high_resolution_clock::now();

    try {
        if (contexts.size() > 1) {
            // Multi-flow broadcast
            log("Using multi-flow broadcast across " + std::to_string(contexts.size()) + " interfaces");

            // Calculate chunk size (capped at maxChunkSizeBytes)
            size_t chunkSize = std::min(maxChunkSizeBytes, fileSize / contexts.size());

            // Calculate number of chunks needed
            size_t numChunks = (fileSize + chunkSize - 1) / chunkSize; // Ceiling division

            log("Splitting " + std::to_string(fileSize / (1024.0 * 1024 * 1024)) +
                "GB of data into " + std::to_string(numChunks) + " chunks (" +
                std::to_string(chunkSize / (1024.0 * 1024 * 1024)) + "GB per chunk)");

            // Calculate chunks per context (some contexts may get one more chunk than others)
            size_t chunksPerContext = numChunks / contexts.size();
            size_t extraChunks = numChunks % contexts.size();

            std::vector<std::exception_ptr> exceptions(contexts.size());
            std::atomic<bool> error_occurred(false);
            std::vector<std::thread> broadcastThreads;

            // Create one thread per context, each handling multiple chunks
            for (size_t contextIdx = 0; contextIdx < contexts.size(); contextIdx++) {
                broadcastThreads.emplace_back([this, contextIdx, chunksPerContext, extraChunks, numChunks, chunkSize, fileSize, fileData, &exceptions, &error_occurred]() {
                    try {
                        // Calculate how many chunks this context handles
                        size_t myChunkCount = chunksPerContext + (contextIdx < extraChunks ? 1 : 0);

                        // Process all chunks assigned to this context
                        for (size_t i = 0; i < myChunkCount; i++) {
                            // Calculate the chunk index in round-robin fashion
                            // Context 0 gets chunks 0, contexts_size, 2*contexts_size...
                            // Context 1 gets chunks 1, contexts_size+1, ...
                            size_t chunkIdx = contextIdx + (i * contexts.size());
                            if (chunkIdx >= numChunks) break; // Safety check

                            size_t offset = chunkIdx * chunkSize;
                            size_t thisChunkSize = (chunkIdx == numChunks - 1) ? (fileSize - offset) : chunkSize;

                            // Log which thread is handling which chunk
                            std::stringstream ss;
                            ss << "Context " << contextIdx << " broadcasting chunk " << chunkIdx
                               << " (offset: " << offset << ", size: " << thisChunkSize << ")";
                            log(ss.str());

                            // Configure broadcast for this chunk
                            gloo::BroadcastOptions opts(contexts[contextIdx]);
                            opts.setRoot(0);
                            opts.setTag(static_cast<uint64_t>(chunkIdx) + 2); // +2 because tag 1 was used for size
                            opts.setOutput(static_cast<uint8_t*>(fileData) + offset, thisChunkSize);

                            // Execute chunk broadcast
                            gloo::broadcast(opts);
                        }
                    } catch (...) {
                        exceptions[contextIdx] = std::current_exception();
                        error_occurred.store(true);
                    }
                });
            }

            // Wait for all threads to complete
            for (auto& thread : broadcastThreads) {
                thread.join();
            }

            // Check for any exceptions in the broadcast threads
            if (error_occurred.load()) {
                for (size_t i = 0; i < contexts.size(); i++) {
                    if (exceptions[i]) {
                        try {
                            std::rethrow_exception(exceptions[i]);
                        } catch (const std::exception& e) {
                            log("Error in broadcast thread for context " + std::to_string(i) + ": " + e.what());
                        }
                    }
                }

                // Clean up memory
                if (fileData != nullptr) {
                    munmap(fileData, fileSize);
                }
                return nullptr;
            }
        } else {
            // Single-flow broadcast
            log("Using single-flow broadcast");

            gloo::BroadcastOptions opts(contexts[0]);
            opts.setRoot(0);
            opts.setTag(2);
            opts.setOutput(fileData, fileSize);

            gloo::broadcast(opts);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        lastBroadcastDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - broadcastStartTime);
        lastBroadcastBandwidthGBs = (fileSize / (1024.0 * 1024 * 1024)) /
                                    (lastBroadcastDuration.count() / 1000.0);

        log("Broadcast complete in " + std::to_string(lastBroadcastDuration.count()) +
            "ms (" + std::to_string(lastBroadcastBandwidthGBs) + " GB/s)");

        // Set output size
        outDataSize = fileSize;
        return fileData;
    }
    catch (const std::exception& e) {
        log("Error during broadcast: " + std::string(e.what()));

        // Clean up memory
        if (fileData != nullptr) {
            munmap(fileData, fileSize);
        }
        return nullptr;
    }
}

int GlooFileBroadcast::getWorldSize() const {
    return worldSize;
}

// Get last broadcast bandwidth
double GlooFileBroadcast::getLastBroadcastBandwidthGBs() const {
    return lastBroadcastBandwidthGBs;
}

// Log helper
void GlooFileBroadcast::log(const std::string& message) {
    std::cout << "[" << getCurrentTimestamp() << "] Rank " << rank
              << ": " << message << std::endl;
}

// Convert string to vector for store API
std::vector<char> GlooFileBroadcast::stringToVector(const std::string& str) {
    return std::vector<char>(str.begin(), str.end());
}

// Convert vector to string for store API
std::string GlooFileBroadcast::vectorToString(const std::vector<char>& vec) {
    return std::string(vec.data(), vec.size());
}

// Discover network interfaces
std::vector<std::pair<std::string, std::string>> GlooFileBroadcast::discoverNetworkInterfaces() {
    std::vector<std::pair<std::string, std::string>> results;

    log("Discovering network interfaces");

    // Get all interfaces with their IPs in one command to reduce process spawning
    FILE* pipe = popen("ip -o -4 addr show | grep -v '127.0.0.1' | grep -v 'docker' | awk '{print $2,$4}' | cut -d/ -f1", "r");
    if (!pipe) {
        log("Failed to run command to discover network interfaces");
        return results;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        if (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }

        // Split by space
        size_t pos = line.find(' ');
        if (pos != std::string::npos) {
            std::string iface = line.substr(0, pos);
            std::string ip = line.substr(pos + 1);
            results.emplace_back(iface, ip);
        }
    }
    pclose(pipe);

    log("Discovered " + std::to_string(results.size()) + " network interfaces");
    for (const auto& pair : results) {
        log("  - " + pair.first + " (IP: " + pair.second + ")");
    }

    // Limit the number of interfaces (usually 4-8 is optimal)
    const int maxInterfaces = 16;
    if (results.size() > maxInterfaces) {
        results.resize(maxInterfaces);
        log("Limiting to " + std::to_string(maxInterfaces) + " interfaces");
    }

    return results;
}

// Coordinate on interfaces
bool GlooFileBroadcast::coordinateInterfaces() {
    log("Coordinating on number of interfaces to use");

    // First, publish how many interfaces we have
    std::string interfaceCountKey = prefix + "_num_interfaces_rank_" + std::to_string(rank);
    store->set(interfaceCountKey, stringToVector(std::to_string(usableInterfaces.size())));

    // Wait for all ranks to publish their interface counts
    log("Waiting for all ranks to publish their interface counts");
    std::vector<int> rankInterfaceCounts(worldSize);
    rankInterfaceCounts[rank] = usableInterfaces.size();

    for (int r = 0; r < worldSize; r++) {
        if (r == rank) continue;

        std::string countKey = prefix + "_num_interfaces_rank_" + std::to_string(r);
        std::string countStr;
        auto startWait = std::chrono::high_resolution_clock::now();

        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            try {
                countStr = vectorToString(store->get(countKey));
            } catch (const std::exception&) {
                countStr = "";
            }

            // Report if waiting too long
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto waitingTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startWait);
            if (waitingTime.count() >= 5 && waitingTime.count() % 5 == 0) {
                log("Still waiting for rank " + std::to_string(r) + " interface count (" +
                    std::to_string(waitingTime.count()) + "s)");
            }
        } while (countStr.empty());

        rankInterfaceCounts[r] = std::stoi(countStr);
    }

    // Find the minimum number of interfaces across all ranks
    int minInterfaces = *std::min_element(rankInterfaceCounts.begin(), rankInterfaceCounts.end());
    log("Minimum interfaces across all ranks: " + std::to_string(minInterfaces));

    // Trim our usableInterfaces to the minimum
    if (usableInterfaces.size() > minInterfaces) {
        usableInterfaces.resize(minInterfaces);
        log("Limiting to " + std::to_string(minInterfaces) + " interfaces for consistency");
    }

    // Share IPs for each interface
    for (size_t i = 0; i < usableInterfaces.size(); i++) {
        std::string interfaceKey = prefix + "_interface_" + std::to_string(i) + "_rank_" + std::to_string(rank);
        store->set(interfaceKey, stringToVector(usableInterfaces[i].second));
        log("Published interface " + std::to_string(i) + " IP: " + usableInterfaces[i].second);
    }

    // Wait for all ranks to publish their interface IPs
    log("Waiting for all ranks to publish their interface IPs");
    for (int r = 0; r < worldSize; r++) {
        if (r == rank) continue;

        // Get IPs for each interface from remote rank
        for (size_t i = 0; i < minInterfaces; i++) {
            std::string interfaceKey = prefix + "_interface_" + std::to_string(i) + "_rank_" + std::to_string(r);
            std::string ip;
            auto startWait = std::chrono::high_resolution_clock::now();

            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                try {
                    ip = vectorToString(store->get(interfaceKey));
                } catch (const std::exception&) {
                    ip = "";
                }

                // Report if waiting too long
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto waitingTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startWait);
                if (waitingTime.count() >= 5 && waitingTime.count() % 5 == 0) {
                    log("Still waiting for rank " + std::to_string(r) + " interface " +
                        std::to_string(i) + " IP (" + std::to_string(waitingTime.count()) + "s)");
                }
            } while (ip.empty());

            log("Found rank " + std::to_string(r) + " interface " + std::to_string(i) + " IP: " + ip);
        }
    }

    log("All interface IPs discovered");
    return true;
}

// Setup contexts
bool GlooFileBroadcast::setupContexts() {
    log("Setting up contexts for " + std::to_string(usableInterfaces.size()) + " interfaces");

    // Create contexts for each interface
    for (size_t i = 0; i < usableInterfaces.size(); i++) {
        log("Creating context for interface " + std::to_string(i) + " (" +
            usableInterfaces[i].first + ", " + usableInterfaces[i].second + ")");

        try {
            // Create device for this interface
            gloo::transport::tcp::attr attr;
            attr.iface = usableInterfaces[i].first;
            attr.hostname = usableInterfaces[i].second;

            auto device = gloo::transport::tcp::CreateDevice(attr);

            // Create context for this interface
            auto context = std::make_shared<gloo::rendezvous::Context>(rank, worldSize);

            // Signal readiness to connect for this interface
            std::string readyKey = prefix + "_ready_iface_" + std::to_string(i) + "_rank_" + std::to_string(rank);
            store->set(readyKey, stringToVector("true"));

            // Wait for all ranks to be ready for connection
            log("Waiting for all ranks to be ready for interface " + std::to_string(i) + " connection");
            for (int r = 0; r < worldSize; r++) {
                if (r == rank) continue;

                std::string remoteReadyKey = prefix + "_ready_iface_" + std::to_string(i) + "_rank_" + std::to_string(r);
                std::string readyVal;
                auto startWait = std::chrono::high_resolution_clock::now();

                do {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    try {
                        readyVal = vectorToString(store->get(remoteReadyKey));
                    } catch (const std::exception&) {
                        readyVal = "";
                    }

                    // Report if waiting too long
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    auto waitingTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startWait);
                    if (waitingTime.count() >= 5 && waitingTime.count() % 5 == 0) {
                        log("Still waiting for rank " + std::to_string(r) + " to be ready for interface " +
                            std::to_string(i) + " (" + std::to_string(waitingTime.count()) + "s)");
                    }
                } while (readyVal != "true");
            }

            log("All ranks ready for interface " + std::to_string(i) + " connection");

            // Add a staggered delay to prevent connection storms
            std::this_thread::sleep_for(std::chrono::milliseconds(rank * 20 + i * 10));

            // Connect to the mesh for this interface
            log("Connecting interface " + std::to_string(i) + " to the mesh");
            auto connectStore = std::make_shared<gloo::rendezvous::PrefixStore>(
                prefix + "_conn_iface_" + std::to_string(i), store);

            context->connectFullMesh(connectStore, device);
            contexts.push_back(context);
            log("Successfully connected interface " + std::to_string(i) + " to the mesh");
        } catch (const std::exception& e) {
            log("Failed to connect interface " + std::to_string(i) + " to the mesh: " + std::string(e.what()));
        }
    }

    if (contexts.empty()) {
        log("ERROR: No interfaces could be connected");
        return false;
    }

    log("Successfully connected " + std::to_string(contexts.size()) + " interfaces");
    return true;
}

std::string GlooFileBroadcast::getUniqueBarrierName(const std::string& baseName, const std::string& filePath) const {
    if (filePath.empty()) {
        return baseName;
    }

    // Extract just the filename part to keep keys manageable
    size_t lastSlash = filePath.find_last_of("/\\");
    std::string fileName = (lastSlash != std::string::npos) ?
                           filePath.substr(lastSlash + 1) :
                           filePath;

    // Return combined name
    return baseName + "_" + fileName;
}

bool GlooFileBroadcast::executeBarrier(const std::string& barrierName) {
    log("Executing barrier: " + barrierName);

    // Simplify the barrier to match the working code
    if (rank == 0) {
        // Only rank 0 sets the barrier key
        std::string barrierKey = prefix + "_barrier_" + barrierName;
        store->set(barrierKey, stringToVector("ready"));
        log("Set barrier key: " + barrierKey);
    }

    // All ranks (including 0) wait for the barrier key
    if (rank == 0) {
        // Give a short delay for the key to propagate in Redis
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::string barrierKey = prefix + "_barrier_" + barrierName;
    std::string value;
    log("Waiting for barrier: " + barrierName);

    auto startWait = std::chrono::high_resolution_clock::now();
    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        try {
            value = vectorToString(store->get(barrierKey));
        } catch (const std::exception&) {
            value = "";
        }

        // Report if waiting too long
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto waitingTime = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startWait);
        if (waitingTime.count() >= 5 && waitingTime.count() % 5 == 0) {
            log("Still waiting for barrier " + barrierName + " (" + std::to_string(waitingTime.count()) + "s)");
        }
    } while (value != "ready");

    log("Passed barrier: " + barrierName);
    return true;
}