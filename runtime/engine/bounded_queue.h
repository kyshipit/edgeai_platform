/*
线程安全的有界阻塞队列，用于流水线阶段之间的帧传递：
容量限制：防止生产者跑太快撑爆内存。
阻塞 push/pop：队列满时生产者等待，空时消费者等待。
支持超时（用于优雅退出）。

使用场景：
pre_queue：存放预处理完成的帧，等待推理线程。
post_queue：存放推理结果，等待后处理/显示。

*/


// engine/bounded_queue.h (header-only（模板类）)
// engine/bounded_queue.h
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    void Push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    T Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    bool TryPop(T& item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [this] { return !queue_.empty(); }))
            return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    size_t capacity_;
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
};