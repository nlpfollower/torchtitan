#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    // Add a task to the queue and get a future for the result
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    // Wait for all tasks to complete
    void wait_all();
    
    // Get count of active tasks
    size_t active_tasks() const;

private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization
    mutable std::mutex queue_mutex;
    std::condition_variable task_condition;
    std::condition_variable finished_condition;
    std::atomic<bool> stop;
    std::atomic<size_t> active_task_count;
};

// Implementation of template method
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (stop) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }
        
        tasks.emplace([task, this]() {
            active_task_count++;
            (*task)();
            active_task_count--;
            finished_condition.notify_all();
        });
    }
    
    task_condition.notify_one();
    return res;
}