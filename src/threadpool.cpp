#include "threadpool.hpp"

ThreadPool::ThreadPool(size_t numThreads)
    : threads(), tasks(), queue_mutex(), stop(false), mutex_condition()
{
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(std::thread(&ThreadPool::ThreadLoop,this));
    }
}

// Repeatedly executes top task in the queue, waiting while empty, and terminates when stop is set to true
void ThreadPool::ThreadLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !tasks.empty() || stop;
            });
            if (stop) {
                return;
            }
            task = tasks.front();
            tasks.pop();
        }

        task();
    }
}

// Adds new task to the queue, and notifies only one thread to prevent race condition
void ThreadPool::AddTask(const std::function<void()>& task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(task);
    }
    mutex_condition.notify_one();
}

bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        poolbusy = !tasks.empty();
    }
    return poolbusy;
}

void ThreadPool::Stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    mutex_condition.notify_all();
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}
