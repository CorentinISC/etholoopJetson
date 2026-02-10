#include "video_writer.h"

cuda::GpuMat videoconversion_gpu(const Mat& img)
{
    cuda::GpuMat gpuIn(img);
    cuda::GpuMat out;

    if (img.type() == CV_8UC1) {
        cuda::cvtColor(gpuIn, out, COLOR_GRAY2RGB);
    }
    else if (img.type() == CV_16UC1) {
        cuda::GpuMat tmp8;
        gpuIn.convertTo(tmp8, CV_8U, 1.0 / 256.0);
        cuda::cvtColor(tmp8, out, COLOR_GRAY2RGB);
    }
    else {
        return gpuIn;
    }

    return out;
}