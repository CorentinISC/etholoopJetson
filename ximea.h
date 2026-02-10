#ifndef XIMEA_H
#define XIMEA_H

#include <xiApi.h>
#include "opencv2/opencv.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace std::chrono;
using clock_type = std::chrono::steady_clock;

class Ximea {
public:
    Ximea(int exp, string userId, int height, int width,XI_IMG_FORMAT image_format);

    HANDLE hCam = nullptr;
    
    XI_IMG img;
    
    int exposure;
    int H, W;
    VideoWriter output;
    XI_IMG_FORMAT img_format;

    string camUserId;

    void InitXimea();
    void closeXimea();
    Mat get_frame();
    void save_frame(Mat frame);
    HANDLE openCameraByUserId(const string& userId);
    Mat ximeaToMat(const XI_IMG& img);
};

#endif // XIMEA_H