#include "frame_queue.h"
#include "ximea.h"
#include "yolo_trt.h"
#include "video_writer.h"
#include <thread>
#include <chrono>
#include <semaphore.h>

sem_t semYOLOReady;
sem_t semWriterReady;

using namespace cv;
using namespace std;

void drawBoxes(Mat& image, const vector<Box>& boxes,
               int inputW, int inputH, int origW, int origH,
               float padX, float padY, float scale) {

    for(const auto& box : boxes) {
        // 1️⃣ Reconvertir de l’espace 640x640 du réseau vers image originale
        float x = (box.cx - box.bw/2 - padX) / scale;
        float y = (box.cy - box.bh/2 - padY) / scale;
        float w = box.bw / scale;
        float h = box.bh / scale;

        // Limiter aux dimensions de l’image
        int left   = max(0, min(origW-1, int(x)));
        int top    = max(0, min(origH-1, int(y)));
        int right  = max(0, min(origW-1, int(x+w)));
        int bottom = max(0, min(origH-1, int(y+h)));

        rectangle(image, Point(left, top), Point(right, bottom),
                      Scalar(0, 255, 0), 2);

        // Afficher la classe et la confidence
        string label = "ID:" + to_string(box.classId) + 
                            " " + to_string(int(box.conf*100)) + "%";
        int baseline = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        top = max(top, labelSize.height);
        rectangle(image, Point(left, top-labelSize.height),
                      Point(left+labelSize.width, top+baseline),
                      Scalar(0,255,0), FILLED);
        putText(image, label, Point(left, top),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0), 1);
    }
}


void writerThread(FrameQueue& queue,const string& filename, int width, int height, double fps){
    
    VideoWriter writer("appsrc ! queue ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=16000000 insert-sps-pps=true ! h264parse ! matroskamux ! filesink location=" +filename, CAP_GSTREAMER,  0, fps, Size (width, height));

    if (!writer.isOpened())
        throw runtime_error("VideoWriter non ouvert");

    sem_post(&semWriterReady);

    while (true) {
        Frame f = queue.pop();
        if (f.image.empty())
            break;
    
    cuda::GpuMat gpuFrame = videoconversion_gpu(f.image);
	cuda::resize(gpuFrame, gpuFrame, Size(width, height));

	Mat outCPU;
	gpuFrame.download(outCPU);
	writer.write(outCPU);   
    }

    writer.release();
}

void yoloThread(FrameQueue& queue,const string& enginepath){
    
    YoloTRT yolo(enginepath);
    
    // Warmup
    // TODO (?)
    
    sem_post(&semYOLOReady);
    
    while (true) {
        Frame f = queue.pop();
        if (f.image.empty()){
            break;
        }

        Mat input = yolo.preprocess(f.image);

        auto output = yolo.infer(input);

        vector<Box> boxes = yolo.postprocess(output, 0.25f, 0.30f,input.cols, input.rows);
        
        int w = f.image.cols;
        int h = f.image.rows;
        float scale = min(640.f / w, 640.f / h);
        float padX = (640 - w*scale)/2;
        float padY = (640 - h*scale)/2;

        // créer une copie pour affichage
        Mat vis;
        f.image.copyTo(vis); // image originale

        drawBoxes(vis, boxes, 640, 640, w, h, padX, padY, scale);

        imshow("Detections", vis);
        waitKey(1);
    }
}

int main() {

    int framerate = 20;
    XI_IMG_FORMAT img_format = XI_RGB24;
    int width = 1280;
    int height = 1024;
    int nb_frame = 500;
    int exposure = 100000;

    // Input Source
    // Ximea ximea(exposure, "XIMEA_X10",height,width,img_format);
    // ximea.InitXimea();
    VideoCapture cap("/home/nvidia/Desktop/yolo_jetson/X20.mkv");
    if (!cap.isOpened()) {
        cout << "Erreur ouverture vidéo" << endl;
        return -1;
    }


    //FrameQueue queue_writer(100);
    FrameQueue queue_yolo(1);

    sem_init(&semYOLOReady, 0, 0);
    //sem_init(&semWriterReady, 0, 0);

    //thread tWriter(writerThread,ref(queue_writer),"/home/nvidia/Desktop/yolo_jetson/output.mkv",width, height, framerate);
    thread tYolo(yoloThread,ref(queue_yolo),"/home/nvidia/Desktop/yolo_jetson/yolov8s.engine");

    sem_wait(&semYOLOReady);
    //sem_wait(&semWriterReady);

    auto frame_period = milliseconds(1000 / framerate);
    auto next_time = clock_type::now();
    auto sleep = clock_type::now();
    auto t0 = clock_type::now();
    Mat img;
    int lost_frame = 0;

    for(int i = 0; i < nb_frame; i++){
        
        // Récupère la frame
        // Mat img = ximea.get_frame();
        Mat img;
        cap >> img;

        // Ecriture pour la vidéo
        // Frame f1;
        // f1.image = img;
        // if (!queue_writer.try_push(move(f1))) {
        //     cout << "Perte de Frame" << endl;
        //     lost_frame+=1;
        // }

        // Inférence YOLO
        Frame f2;
        f2.image = img;
        queue_yolo.try_push(move(f2));

        // For the framerate
        next_time += frame_period;
        auto sleep_time = next_time - clock_type::now();
        if(sleep_time > milliseconds(0))
            this_thread::sleep_for(sleep_time);
        else
            next_time = clock_type::now();
      
    }
    auto t1 = clock_type::now();
    auto dt = duration<double>(t1 - t0).count();
    float expected = static_cast<float>(nb_frame) / framerate;

    Frame f;
    // queue_writer.push(move(f));
    // tWriter.join();
    queue_yolo.push(move(f));
    tYolo.join();

    //ximea.closeXimea();

    cout << "Temps de la boucle : " << dt << " Prévu : " << expected << endl;
    cout << "Frames perdues : " << lost_frame << endl;
    return 0;
}
