#include "frame_queue.h"

FrameQueue::FrameQueue(size_t maxSize) : maxSize_(maxSize) {};

void FrameQueue::push(Frame&& frame) {
    unique_lock<mutex> lock(mutex_);
    condFull_.wait(lock, [&] { return queue_.size() < maxSize_; });
    queue_.push(move(frame));
    condEmpty_.notify_one();
}

Frame FrameQueue::pop() {
    unique_lock<mutex> lock(mutex_);
    condEmpty_.wait(lock, [&] { return !queue_.empty() || stop_; });

    if (queue_.empty())
        return {};  // frame invalide si arrêt

    Frame f = move(queue_.front());
    queue_.pop();
    condFull_.notify_one();
    return f;
}

bool FrameQueue::try_push(Frame&& frame) {
    lock_guard<mutex> lock(mutex_);

    if (queue_.size() >= maxSize_)
        return false;

    queue_.push(move(frame));
    condEmpty_.notify_one();
    return true;
}


void FrameQueue::stop() {
    lock_guard<mutex> lock(mutex_);
    stop_ = true;
    condEmpty_.notify_all();
}