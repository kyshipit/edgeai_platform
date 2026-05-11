/*
一个固定大小的线程池，支持：
任意返回类型的任务入队（通过 std::future）。
CPU 亲和性绑定（未来每个推理线程绑定到大核）。
优雅停止（析构时 join 所有线程）。
*/


// engine/thread_pool.h
#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <atomic>

class ThreadPool {
public:
    // cpu_affinity: 每个线程绑定的 CPU 核心编号（空 = 不绑定）
    explicit ThreadPool(size_t num_threads,
                        const std::vector<int>& cpu_affinity = {});
    ~ThreadPool();

    // 提交任务，返回 future 用于获取结果
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;

    size_t Pending() const;   // 队列中未执行任务数

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
};

// 模板实现必须放在头文件
template<class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_)
            throw std::runtime_error("Enqueue on stopped ThreadPool");
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return res;
}