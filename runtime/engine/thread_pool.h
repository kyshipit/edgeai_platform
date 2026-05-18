/*
 * engine/thread_pool.h
 *
 * 【engine 层】固定大小线程池，供 Pipeline 启动多个长期运行的 InferenceLoop。
 *
 * - Enqueue 返回 std::future，可获取任务返回值。
 * - 构造时可指定 cpu_affinity，将工作线程绑定到指定 CPU 核。
 * - 析构时 stop_ 置位并 join 所有 worker。
 */
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