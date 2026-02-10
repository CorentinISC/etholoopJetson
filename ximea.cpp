#include "ximea.h"

Ximea::Ximea(int exp, string userId,int height, int width,XI_IMG_FORMAT image_format)
{
    exposure   = exp;
    camUserId  = userId;
    H = height;
    W = width;
    img_format = image_format;
    memset(&img, 0, sizeof(XI_IMG));
    img.size = sizeof(XI_IMG);
}

HANDLE Ximea::openCameraByUserId(const string& target)
{
    DWORD nb = 0;
    xiGetNumberDevices(&nb);

    for (DWORD i = 0; i < nb; ++i) {
        HANDLE h;
        if (xiOpenDevice(i, &h) != XI_OK)
            continue;

        char id[64] = {0};
        xiGetParamString(h, XI_PRM_DEVICE_USER_ID, id, sizeof(id));

        if (target == id) {
            return h;
        }

        xiCloseDevice(h);
    }
    return nullptr;
}

void Ximea::InitXimea()
{
    hCam = openCameraByUserId(camUserId);
    if (!hCam) {
        cout << "XIMEA camera not found: " << camUserId << endl;
        exit(1);
    }

    // Configuration caméra
    xiSetParamInt(hCam, XI_PRM_EXPOSURE, exposure);

    xiSetParamInt(hCam, XI_PRM_GAIN, 0);
    xiSetParamInt(hCam, XI_PRM_AEAG, 0);
    xiSetParamInt(hCam, XI_PRM_IMAGE_DATA_FORMAT, img_format);

    // Prends toujours le max possible pour le framerate
    float framerate_max = 0;
    xiGetParamFloat(hCam, XI_PRM_FRAMERATE XI_PRM_INFO_MAX, &framerate_max);
    xiSetParamInt(hCam, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FRAME_RATE);
    xiSetParamFloat(hCam, XI_PRM_FRAMERATE, framerate_max);

    // Lire taille image réelle

    xiSetParamInt(hCam, XI_PRM_WIDTH,  W);
    xiSetParamInt(hCam, XI_PRM_HEIGHT, H);

    if (xiStartAcquisition(hCam) != XI_OK)
        cout << "Impossible de démarrer acquisition Ximea" << endl;
}

Mat Ximea::get_frame(){
    auto t0 = clock_type::now();
    if (xiGetImage(hCam, 100, &img) != XI_OK)
        _exit(1);
    auto t1 = clock_type::now();
    auto dt = duration<double>(t1 - t0).count();

    cout << "Temps de la capture : " << dt << endl;
    
    return ximeaToMat(img);

}

Mat Ximea::ximeaToMat(const XI_IMG& img){

    switch (img.frm)
    {
        case XI_RAW8:
        case XI_MONO8:
            return Mat(img.height, img.width, CV_8UC1, img.bp).clone();

        case XI_MONO16:
            return Mat(img.height, img.width, CV_16UC1, img.bp).clone();

        case XI_RGB24:
            return Mat(img.height, img.width, CV_8UC3, img.bp).clone();

        case XI_RAW16:
            return Mat(img.height, img.width, CV_16UC1, img.bp).clone();

        default:
            throw runtime_error("Pixel format XIMEA non supporté");

    }
}

void Ximea::closeXimea(){

    xiStopAcquisition(hCam);
    xiCloseDevice(hCam);
    output.release();

    cout << "XIMEA END (" << camUserId << ")" << endl;
}