#include "thread_pool.h"

ThreadPool::ThreadPool(size_t threads) : stop(false), active_task_count(0) {
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);

                    task_condition.wait(lock, [this] {
                        return stop || !tasks.empty();
                    });

                    if (stop && tasks.empty()) {
                        return;
                    }

                    task = std::move(tasks.front());
                    tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop = true;
    }

    task_condition.notify_all();

    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex);

    finished_condition.wait(lock, [this] {
        return tasks.empty() && active_task_count.load() == 0;
    });
}

size_t ThreadPool::active_tasks() const {
    return active_task_count.load();
}