#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;

cuda::GpuMat videoconversion_gpu(const Mat& img);