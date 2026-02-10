#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <cuda_fp16.h>

using namespace std;
using namespace cv;

struct Box {
    float cx, cy, bw, bh;
    float conf;
    int classId;
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class YoloTRT {
public:

    Logger gLogger;    
    YoloTRT(const string& enginePath);
    vector<float> infer(const Mat& input);
    Mat preprocess(const Mat& input);
    vector<Box> postprocess(const vector<float>& output,float confThresh,float nmsThresh,int imgW,int imgH);
    int numClasses;
    int inputIndex;
    int outputIndex;

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    void* buffers[2];
    cudaStream_t stream;
};