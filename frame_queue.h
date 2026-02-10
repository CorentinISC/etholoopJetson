#include <queue>
#include <mutex>
#include <condition_variable>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

struct Frame {
    Mat image;
};

class FrameQueue {
public:
    FrameQueue(size_t maxSize);
    void push(Frame&& frame);
    Frame pop();
    void stop();
    bool try_push(Frame&& frame);
private:
    queue<Frame> queue_;
    mutex mutex_;
    condition_variable condEmpty_, condFull_;
    size_t maxSize_;
    bool stop_ = false;
};