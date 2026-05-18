/*
 * engine/thread_pool.cpp
 * 线程池 worker 实现：可选 CPU 亲和性 + 任务队列消费循环。
 */
#include "thread_pool.h"
#include <pthread.h>

ThreadPool::ThreadPool(size_t num_threads,
                       const std::vector<int>& cpu_affinity) : stop_(false) {
    std::vector<int> affinity = cpu_affinity;
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this, i, affinity]() {
            if (!affinity.empty() && i < affinity.size()) {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(affinity[i], &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            }
            // 工作循环
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });
                    if (stop_ && tasks_.empty())
                        return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable())
            worker.join();
    }
}

size_t ThreadPool::Pending() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}