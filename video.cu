/*
                                        Ali NouriZonoz
                                        Huberlab
                                        Department of Basic Neuroscience
                                        University of Geneva
                                        Switzerland

                                        Lucas Maigre
                                        Institut des Sciences Cognitives
                                        Bron
                                        France


                                        2D extraction of a single LED from XIMEA camera using Jetson TX2

*/

#include "configuration.h"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/photo/cuda.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

/******************************************************************
 * GLOBALS
 ******************************************************************/

VideoCapture cap;
string idCam;

atomic<bool> stopRecording(false);
atomic<bool> startRecord(false);

/******************************************************************
 * UDP GLOBALS
 ******************************************************************/
int sockfd;
sockaddr_in pcAddr;
bool startRequest = false;

/******************************************************************
 * FRAME PACKET
 ******************************************************************/
struct FramePacket
{
    cv::Mat raw;
    int64_t timestamp_ms;  // steady_clock timestamp in ns
};

mutex qMutex;
condition_variable qCv;
deque<FramePacket> queueFrames;

/******************************************************************
 * CUDA BUFFERS (encoder thread local use)
 ******************************************************************/
cuda::GpuMat gpu_raw;
cuda::GpuMat gpu_rgb_uncorrected;
cuda::GpuMat gpu_rgb;

/******************************************************************
 * GPU processing
 ******************************************************************/
static void gpu_getRGBimage(const cuda::GpuMat& gpu_imageRAW, cuda::GpuMat& gpu_imageRGB)
{
    cuda::demosaicing(gpu_imageRAW, gpu_imageRGB, COLOR_BayerRG2BGR);
    cuda::multiply(gpu_imageRGB, Scalar(WB_BLUE, WB_GREEN, WB_RED), gpu_imageRGB);
}

/******************************************************************
 * UDP communication
 ******************************************************************/
/* createSocket
 * function creating a socket to communicate via UDP with the hostmachine
 */
int createSocket(char* hostMachine, char* port)
{
    sockaddr_in localAddr{};

    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    int yes = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));

    // Adresse locale : écouter partout
    localAddr.sin_family = AF_INET;
    localAddr.sin_port = htons(BROADCAST_PORT);
    localAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sockfd, (struct sockaddr*)&localAddr, sizeof(localAddr)) < 0) {
        perror("bind");
        close(sockfd);
        return -1;
    }

    // Adresse du PC (unicast)
    memset(&pcAddr, 0, sizeof(pcAddr));
    pcAddr.sin_family = AF_INET;
    pcAddr.sin_port = htons(atoi(port));

    if (inet_aton(hostMachine, &pcAddr.sin_addr) == 0) {
        perror("inet_aton");
        close(sockfd);
        return -1;
    }
    return 0;
}

/* sendUnicastReply
 * function to answer broadcast messages only to hostmachine
 */
void sendUnicastReply(const char* msg, const sockaddr_in& dest)
{
    sendto(sockfd,
           msg,
           strlen(msg),
           0,
           (const sockaddr*)&dest,
           sizeof(dest));
}

/* udpSocket
 * Thread for the communication between jetson and the hostmachine
 */
void udpThread()
{
    char buf[50];
    sockaddr_in srcAddr{};
    socklen_t srcLen = sizeof(srcAddr);

    while (!stopRecording) {

        int n = recvfrom(sockfd,
                         buf,
                         sizeof(buf) - 1,
                         0,
                         (struct sockaddr*)&srcAddr,
                         &srcLen);

        if (n <= 0)
            continue;

        buf[n] = '\0';

        std::cout << "Received: " << buf << std::endl;

        // Traitement des messages
        if (strcmp(buf, START_MESSAGE) == 0) {
            startRequest = true;
        }
        else if (strcmp(buf, STOP_MESSAGE) == 0) {
            stopRecording = true;
	    qCv.notify_all();
        }
    }
}

/******************************************************************
 * CAMERA THREAD
 ******************************************************************/
void captureThread(VideoCapture* cap)
{
    while (!stopRecording)
    {
        if (!cap->grab())
        {
            cerr << "[CAPTURE] grab failed" << endl;
            continue;
        }

        cv::Mat raw;
        cap->retrieve(raw);

        if (raw.empty())
            continue;

        int64_t ts_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

        {
            lock_guard<mutex> lock(qMutex);

            // drop oldest if full
            if (queueFrames.size() >= MAX_QUEUE_SIZE)
                queueFrames.pop_front();

            queueFrames.push_back(FramePacket{raw.clone(), ts_ms});
        }

        qCv.notify_one();
    }
}

/******************************************************************
 * GStreamer helper
 ******************************************************************/
static GstElement* buildPipeline(const string& filename, int width, int height)
{
    // appsrc will provide BGR frames (CPU)
    // nvvidconv converts to NVMM
    // nvv4l2h265enc does hardware encoding
    // matroskamux writes mkv

    string pipelineStr =
        "appsrc name=mysrc is-live=true format=time do-timestamp=false block=true caps=video/x-raw,format=BGR,width=" +
        to_string(width) + ",height=" + to_string(height) + ",framerate=" + to_string(TARGET_FPS) + "/1 ! "
        "queue max-size-buffers=10 leaky=downstream ! "
        "videoconvert ! "
        "video/x-raw,format=BGRx ! "
        "nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        "nvv4l2h265enc bitrate=" + to_string(BITRATE) + " iframeinterval=" + to_string(IFRAMEINTERVAL) + " insert-sps-pps=true preset-level=1 maxperf-enable=1 ! "
        "h265parse ! "
        "matroskamux ! "
        "filesink location=" + filename + " sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(pipelineStr.c_str(), &error);

    if (!pipeline)
    {
        cerr << "[GSTREAMER] Failed to create pipeline: " << error->message << endl;
        g_error_free(error);
        return nullptr;
    }

    return pipeline;
}

/******************************************************************
 * RECORD THREAD (encoder)
 ******************************************************************/
void encoderThread()
{
    unsigned int width  = cap.get(CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    string outDir = OUTPUT_DIRECTORY;
    string videoFilename = outDir + idCam + ".mkv";
    string tsFilename    = outDir + idCam + "_timestamps.txt";

    ofstream timestampsFile(tsFilename, ios::out | ios::trunc);
    if (!timestampsFile.is_open())
    {
        cerr << "[ENCODER] Cannot open timestamps file" << endl;
        stopRecording = true;
        return;
    }

    gst_init(nullptr, nullptr);

    GstElement* pipeline = buildPipeline(videoFilename, width, height);
    if (!pipeline)
    {
        stopRecording = true;
        return;
    }

    GstElement* appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
    if (!appsrc)
    {
        cerr << "[ENCODER] Failed to retrieve appsrc element" << endl;
        gst_object_unref(pipeline);
        stopRecording = true;
        return;
    }

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    cv::Mat bgr_cpu;

    // Reference timestamp to make PTS start at 0
    int64_t first_ts_ms = -1;

    // For strict FPS output
    int64_t next_expected_ts_ms = -1;

    while (!stopRecording)
    {
        FramePacket pkt;

        {
            unique_lock<mutex> lock(qMutex);

            qCv.wait(lock, [] {
                return stopRecording || !queueFrames.empty();
            });

            if (stopRecording)
                break;

            pkt = std::move(queueFrames.front());
            queueFrames.pop_front();
        }

        if (!startRecord)
            continue;

        if (first_ts_ms < 0)
        {
            first_ts_ms = pkt.timestamp_ms;
            next_expected_ts_ms = pkt.timestamp_ms;
        }

        // Drop frames if camera is faster than target fps
        if (pkt.timestamp_ms < next_expected_ts_ms)
        {
            continue;
        }
        next_expected_ts_ms += FRAME_PERIOD_MS;

        // GPU processing
        gpu_raw.upload(pkt.raw);
        gpu_getRGBimage(gpu_raw, gpu_rgb_uncorrected);
        cuda::gammaCorrection(gpu_rgb_uncorrected, gpu_rgb);
        gpu_rgb.download(bgr_cpu);

        // Allocate gst buffer
        size_t dataSize = bgr_cpu.total() * bgr_cpu.elemSize();
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, dataSize, nullptr);

        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, bgr_cpu.data, dataSize);
        gst_buffer_unmap(buffer, &map);

        // Set timestamps
        int64_t pts_ms = pkt.timestamp_ms - first_ts_ms;

        GST_BUFFER_PTS(buffer) = pts_ms;
        GST_BUFFER_DTS(buffer) = pts_ms;
        GST_BUFFER_DURATION(buffer) = FRAME_PERIOD_MS;

        // Log acquisition timestamp (real)
        timestampsFile << pkt.timestamp_ms << "\n";

        // Push to pipeline
        GstFlowReturn ret;
        g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);

        if (ret != GST_FLOW_OK)
        {
            cerr << "[ENCODER] push-buffer failed, stopping" << endl;
            stopRecording = true;
            break;
        }
    }

    // End stream cleanly
    g_signal_emit_by_name(appsrc, "end-of-stream", nullptr);

    GstBus* bus = gst_element_get_bus(pipeline);
    gst_bus_timed_pop_filtered(bus,GST_CLOCK_TIME_NONE,(GstMessageType)(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    gst_object_unref(bus);

    gst_element_set_state(pipeline, GST_STATE_NULL);

    timestampsFile.close();

    cout << "[ENCODER] stopped cleanly" << endl;
}

/******************************************************************
 * CAMERA INIT
 ******************************************************************/
int cameraInit(VideoCapture& cap)
{
    int open = cap.open(CV_CAP_XIAPI);
    if (!open)
    {
        cout << "ERROR : Ximea camera can not be opened." << endl;
        return 0;
    }

    cap.set(CV_CAP_PROP_XI_GAIN, ETH_GAIN);
    cap.set(CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE, ETH_SENSOR_MODE);
    cap.set(CV_CAP_PROP_XI_DATA_FORMAT, ETH_DATA_FORMAT);
    cap.set(CV_CAP_PROP_XI_AEAG, ETH_AUTOMATIC_EXPOSURE_AND_GAIN);
    cap.set(CV_CAP_PROP_XI_EXPOSURE, ETH_EXPOSURE_TIME);
    cap.set(CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH, ETH_OUTPUT_DATA_BIT_DEPTH);
    cap.set(CV_CAP_PROP_XI_AUTO_WB, ETH_AUTOMATIC_WHITE_BALANCE);
    cap.set(CV_CAP_PROP_XI_GAMMAY, ETH_GAMMAY);

    cap.set(CV_CAP_PROP_XI_ACQ_TIMING_MODE, 1);

    // Set camera fps
    cap.set(CV_CAP_PROP_XI_FRAMERATE, TARGET_FPS);

    cout << "[PARAMETERS] Gain - Wanted : " << ETH_GAIN << " | Set : " << cap.get(CV_CAP_PROP_XI_GAIN) << endl;
    cout << "[PARAMETERS] FPS - Wanted : " << TARGET_FPS << " | Set : " << cap.get(CV_CAP_PROP_XI_FRAMERATE) << endl;
    cout << "[PARAMETERS] Sensor Mode - Wanted : " << ETH_SENSOR_MODE << " | Set : " << cap.get(CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE) << endl;
    cout << "[PARAMETERS] Data Format - Wanted : " << ETH_DATA_FORMAT << " | Set : " << cap.get(CV_CAP_PROP_XI_DATA_FORMAT) << endl;
    cout << "[PARAMETERS] Automatic exposure and gain - Wanted : " << ETH_AUTOMATIC_EXPOSURE_AND_GAIN << " | Set : " << cap.get(CV_CAP_PROP_XI_AEAG) << endl;
    cout << "[PARAMETERS] Exposure time - Wanted : " << ETH_EXPOSURE_TIME << " | Set : " << cap.get(CV_CAP_PROP_XI_EXPOSURE) << endl;
    cout << "[PARAMETERS] Output Data Bit Depth - Wanted : " << ETH_OUTPUT_DATA_BIT_DEPTH << " | Set : " << cap.get(CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH) << endl;
    cout << "[PARAMETERS] Auto White Balance - Wanted : " << ETH_AUTOMATIC_WHITE_BALANCE << " | Set : " << cap.get(CV_CAP_PROP_XI_AUTO_WB) << endl;
    cout << "[PARAMETERS] GammaY - Wanted : " << ETH_GAMMAY << " | Set : " << cap.get(CV_CAP_PROP_XI_GAMMAY) << endl;

    return 1;
}

/******************************************************************
 * MAIN
 ******************************************************************/
int main(int argc, char* argv[])
{
    // get the chosen mode, it's the second argument when calling the program
    string mode = string(argv[1]);
    // variables for the udp socket
    char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
    char* port = argv[3];           // the fourth variable is the port
    
    cout << "[PARAMETERS] Selected mode : " << mode << endl;
    cout << "[PARAMETERS] Name of host machine : " << hostMachine << endl;
    cout << "[PARAMETERS] Port of host machine : " << port << endl;
    cout << "[PARAMETERS] Video Framerate : " << TARGET_FPS << endl;
    cout << "[PARAMETERS] Frame period : " << FRAME_PERIOD_MS << endl;
    cout << "[PARAMETERS] Max Queue Size : " << MAX_QUEUE_SIZE << endl;
    cout << "[PARAMETERS] Bitrate : " << BITRATE << endl;
    cout << "[PARAMETERS] IFRAMEINTERVAL : " << IFRAMEINTERVAL << endl;
    cout << "[PARAMETERS] WB_BLUE : " << WB_BLUE << endl;
    cout << "[PARAMETERS] WB_GREEN : " << WB_GREEN << endl;
    cout << "[PARAMETERS] WB_RED : " << WB_RED << endl;

    char message[50];

    char hostname[50];
    gethostname(hostname, HOST_NAME_MAX);
    idCam = hostname;
    transform(idCam.begin(), idCam.end(), idCam.begin(), ::toupper);


    if(mode == "record")
    {
	cout << "Initializing Ximea Camera." << endl;
        if (!cameraInit(cap))
        {
            cerr << "Camera init failed." << endl;
            return 0;
        }

        // create socket
        cout << "Creating UDP socket." << endl;
        createSocket(hostMachine, port);

        cout << "Starting capture/encode pipeline." << endl;

        // Start threads
	std::thread tCom(udpThread);
        std::thread tCap(captureThread, &cap);
        std::thread tEnc(encoderThread);

        // Send a message to inform the system is ready.
	strcpy(message, READY_MESSAGE);
        sendto(sockfd, &message, sizeof(message), 0, (struct sockaddr*)&pcAddr, sizeof(pcAddr));

        cout<<"Ready message sent. Waiting for the start signal."<<endl;
        
        while(!startRequest){}	// TODO : Add a timeout to end the process.

        cout<<"Start message received."<<endl;

	startRecord = true;
	qCv.notify_all();

	tCom.join();
	tCap.join();
        tEnc.join();

        cap.release();
    }

    cout << "End of Jetson script." << endl;
    return 0;
}