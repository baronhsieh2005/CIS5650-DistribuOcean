#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <vector>
#include <mutex>

class ThreadPool {
private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    bool stop;
    std::condition_variable mutex_condition;
public:
    void ThreadLoop();
    bool busy();
    void AddTask(const std::function<void()>& task);
    void Stop();
    ThreadPool(size_t numThreads);
};
