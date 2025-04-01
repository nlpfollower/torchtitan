#include "tensor_common.h"

// Current log level
static LogLevel current_log_level = INFO;

// Mutex for thread-safe console output
static std::mutex cout_mutex;

// Thread-safe logger implementation
void log(LogLevel level, const std::string& message) {
    if (level >= current_log_level) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        const char* level_str = "";
        switch (level) {
            case DEBUG:   level_str = "DEBUG"; break;
            case INFO:    level_str = "INFO"; break;
            case WARNING: level_str = "WARNING"; break;
            case ERROR:   level_str = "ERROR"; break;
        }

        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char time_buf[20];
        std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&now_c));

        std::cout << "[" << time_buf << "][PRELOADER][" << level_str << "] " << message << std::endl;
    }
}

// Set the log level
void set_log_level(const std::string& level) {
    if (level == "DEBUG") {
        current_log_level = DEBUG;
    } else if (level == "INFO") {
        current_log_level = INFO;
    } else if (level == "WARNING") {
        current_log_level = WARNING;
    } else if (level == "ERROR") {
        current_log_level = ERROR;
    }
}

// Generate a unique key for a filepath
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