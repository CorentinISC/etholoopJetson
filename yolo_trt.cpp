#include "yolo_trt.h"

void Logger::log(Severity severity, const char* msg) noexcept
    {
        // Ignore info-level messages
        if (severity != Severity::kINFO)
            cout << msg << endl;
    }

YoloTRT::YoloTRT(const string& enginePath)
{  
    numClasses = 80;
    ifstream file(enginePath, ios::binary);
    file.seekg(0, ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, ifstream::beg);

    vector<char> engineData(size);
    file.read(engineData.data(), size);

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    cudaStreamCreate(&stream);

    // allocation buffers
    size_t inputSize  = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize);

    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i))
            inputIndex = i;
        else
            outputIndex = i;
    }

    context->setBindingDimensions(inputIndex, nvinfer1::Dims4(1, 3, 640, 640));

    if (!context->allInputDimensionsSpecified())
        throw runtime_error("Input dimensions not set");
    
}


// ----------------- Preprocess -----------------
Mat YoloTRT::preprocess(const Mat& input)
{

    // Convert BGR -> RGB
    Mat rgb;
    cvtColor(input, rgb, COLOR_BGR2RGB);

    // Letterbox pour garder le ratio original
    int w = input.cols;
    int h = input.rows;
    float scale = min(640.f / w, 640.f / h);
    int newW = int(w * scale);
    int newH = int(h * scale);

    Mat resized;
    resize(rgb, resized, Size(newW, newH));

    Mat blob = Mat::zeros(640, 640, CV_32FC3);

    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    // Centrer l'image
    int dx = (640 - newW) / 2;
    int dy = (640 - newH) / 2;
    resized.copyTo(blob(Rect(dx, dy, newW, newH)));

    return blob;
}

// ----------------- Infer -----------------
vector<float> YoloTRT::infer(const Mat& input)
{
    // Input CHW
    vector<float> inputCHW(3 * 640 * 640);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 640; h++)
            for (int w = 0; w < 640; w++)
                inputCHW[c * 640 * 640 + h * 640 + w] = input.at<Vec3f>(h, w)[c];

    cudaMemcpyAsync(
        buffers[inputIndex],
        inputCHW.data(),
        3 * 640 * 640 * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );

    context->enqueueV2(buffers, stream, nullptr);

    vector<float> output(84 * 8400);
    cudaMemcpyAsync(
        output.data(),
        buffers[outputIndex],
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream);
    return output;
}

// ----------------- Postprocess -----------------
vector<Box> YoloTRT::postprocess(const vector<float>& output,float confThresh,float nmsThresh,int imgW,int imgH) {
    
    vector<Box> boxes;
    vector<Rect> rects;
    vector<float> scores;

    const int numBoxes = 8400;

    for (int i = 0; i < numBoxes; i++) {
        float cx = output[0*numBoxes + i];
        float cy = output[1*numBoxes + i];
        float bw = output[2*numBoxes + i];
        float bh = output[3*numBoxes + i];
        float objScore = output[4*numBoxes + i];

        // classes

        float bestClsScore = 0.f;
        int classId = -1;

        for(int c=0;c<numClasses;c++)
        {
            float score = output[(4+c)*numBoxes + i];
            if(score > bestClsScore){
                bestClsScore = score;
                classId = c;
            }
        }

        float conf = bestClsScore;

        if(conf < confThresh) continue;

        float x = (cx - bw/2) * imgW;
        float y = (cy - bh/2) * imgH;
        float w = bw * imgW;
        float h = bh * imgH;

        rects.emplace_back(x,y,w,h);

        scores.push_back(conf);

        boxes.push_back({cx,cy,bw,bh,conf,classId});
    }

    vector<int> keep;
    dnn::NMSBoxes(rects, scores, confThresh, nmsThresh, keep);

    vector<Box> finalBoxes;
    for(int idx : keep)
        finalBoxes.push_back(boxes[idx]);

    return finalBoxes;
}