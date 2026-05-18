/*
 * engine/bounded_queue.h
 *
 * 【engine 层】线程安全有界阻塞队列，用于流水线阶段间传递 InferenceTask。
 *
 * - 容量上限：防止预处理过快撑爆内存（背压）。
 * - Push 满则阻塞，Pop 空则阻塞。
 * - TryPop：带超时，可用于优雅退出（当前 Pipeline 主要用哨兵任务 frame_id==-1）。
 *
 * Pipeline 中的用法：
 * - infer_queues_[i]：预处理线程 → 第 i 个推理线程
 * - post_queue_：推理线程 → 后处理/显示线程
 */
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

    bool TryPush(T item, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_full_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                [this] { return queue_.size() < capacity_; })) {
            return false;
        }
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
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