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

#include <atomic>
#include <netdb.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <chrono>
#include <algorithm>
#include <sys/types.h> 
#include <sys/socket.h>
#include <arpa/inet.h>



/****************************************************************    Defining Global Variables        ***********************************************************/

using namespace cv;
using namespace std;
using namespace std::chrono;

VideoCapture cap;                   // gets the images with opencv
atomic<bool> getFrame;              // boolean telling when the camera frame is captured
Mat imageDenoiseSelected;           // denoised image
Mat imageRAW, imageRGB, imageSHOWN; // images matrices
cuda::GpuMat    gpu_imageRAW,       // gpu matrix RAW image
gpu_imageRGB_uncorrected,
gpu_imageRGB,                       // gpu matrix RGB image
gpu_imageHSV;                       // gpu matrix HSV image
cuda::GpuMat 	gpu_image[4], gpu_image_unoised[4];

bool startRecord = false;
bool okMsg = false;

// create thread for capturing the image
pthread_t thread_cam, thread_record, thread_udp;

string idCam;

bool stopRecording = false;
long int timestamp = 0;

// variables for udp connection
int sockfd;
sockaddr_in pcAddr; 
addrinfo *p = new addrinfo();





/* gpu_getRGBimage
 * function returning RGB image from RAW captured image
 */
void gpu_getRGBimage(cuda::GpuMat gpu_imageRAW, cuda::GpuMat& gpu_imageRGB){

    // demosaicing RAW image
    cuda::demosaicing(gpu_imageRAW, gpu_imageRGB, COLOR_BayerRG2BGR);
    // multiplying with RGB scalar for the white balance
    cuda::multiply(gpu_imageRGB, Scalar(WB_BLUE, WB_GREEN, WB_RED), gpu_imageRGB);
}


/* gpu_getHSVimage
 * function returning HSV image from RGB image
 */
void gpu_getHSVimage(cuda::GpuMat gpu_imageRGB, cuda::GpuMat& gpu_imageHSV){

    // convert image from RGB to HSV
    cuda::cvtColor(gpu_imageRGB, gpu_imageHSV, COLOR_BGR2HSV);
}

// get gpu matrices (RGB, HSV) from the cpu matrix
void getImageRGB(){

    //store in gpu variable the RAW, RGB and HSV image
    gpu_imageRAW.upload(imageRAW);
    gpu_getRGBimage(gpu_imageRAW, gpu_imageRGB_uncorrected);

    cuda::gammaCorrection(gpu_imageRGB_uncorrected, gpu_imageRGB);
    //gpu_imageRGB = gpu_imageRGB_uncorrected;
    gpu_getHSVimage(gpu_imageRGB, gpu_imageHSV);
}

/****************************************************************************************************************************************************************/




/*************************************************************    CPU Thread for capturing images        ********************************************************/

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
void* udpReceiverThread(void*)
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
            okMsg = true;
        }
        else if (strcmp(buf, STOP_MESSAGE) == 0) {
            stopRecording = true;
        }
    }
    return nullptr;
}


/* getImage
 * Thread capturing the camera frame and storing it in imageRAW
 */

void * getImage(void *input)
{
    // capture opencv variable
    VideoCapture *cap = (VideoCapture*) input;

    // capture the frame until the user exists the program
    while(!stopRecording){
        int a = cap->grab();
        cap->retrieve(imageRAW);

        // if the captured frame is not empty, set getFrame flag to true
        if(!imageRAW.empty())
	{
	    timestamp = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();
	    getFrame.store(true);
	}
        if(!a)
            return NULL;
    }

    return NULL;
}

void * recordCam(void *input){

    // Variables used to ensure frame rate correctness
    long int last_frame_time = 0, actual_time = 0;

    // Get resolution and framerate from capture
    unsigned int width = cap.get (cv::CAP_PROP_FRAME_WIDTH);
    unsigned int height = cap.get (cv::CAP_PROP_FRAME_HEIGHT);
    unsigned int fps = ETH_FRAMERATE;
    float intervalFrameMicrosec = 1000000/fps;

    cout<< idCam << " camera configuration :" <<endl;

    cout <<"   - height: " << height << endl;
    cout <<"   - width: " << width << endl;
    cout <<"   - framerate: " << fps << endl;

    // video directory
    string outDir = OUTPUT_DIRECTORY;

    // gpu image of mask
    cuda::GpuMat gpu_image_convert(height, width , 16);

    // Create the writer with gstreamer pipeline encoding into H265, muxing into mkv container and saving to file
    VideoWriter gst_nvh265_writer(
  "appsrc ! queue ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=BGRx ! "
  "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
  "nvv4l2h265enc bitrate=3000000 iframeinterval=30 insert-sps-pps=true ! "
  "h265parse ! matroskamux ! filesink location=" + outDir + idCam + ".mkv sync=true",
  CAP_GSTREAMER,
  0, fps, cv::Size(width, height)
);

    // Create a txt file to save frames timestamps
    ofstream timestampsFile;
    timestampsFile.open(outDir + idCam + "_timestamps.txt", std::ofstream::out | std::ofstream::trunc);

    if (!gst_nvh265_writer.isOpened ()) {
        std::cout << "ERROR : Failed to open gst_nvh265 writer." << std::endl;
        return (NULL);
    }

    // image RGB + image mask
    cuda::GpuMat gpu_imageConcat(height, width, 16);
    Mat imageConcat;

    while(!stopRecording){
	// Wait for the start signal
        if(!startRecord)
            continue;

        // Compute actual time in microseconds
        actual_time = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();

        if(actual_time - last_frame_time >= intervalFrameMicrosec ){
	    // Save the actual time for the next iteration
            last_frame_time = duration_cast< microseconds >( system_clock::now().time_since_epoch() ).count();
	    // Save in the timestamps file the exact moment when the frame has been acquired.
	    timestampsFile << timestamp << std::endl;
            // Conversion from GPU matrix to CPU matrix
            gpu_imageRGB.download(imageRGB);
            // Write the CPU matrix (the frame) in the video file.
            gst_nvh265_writer.write(imageRGB);
        }
    }

    // Release the video writer token.
    gst_nvh265_writer.release();
    cout<<"Video writer released."<<endl;
    // Close the timestamps file.
    timestampsFile.close();

    return NULL;
}


/**********************************************************************    CPU functions        ******************************************************************/




/* cameraInit
 * Function initializing camera parameters
 */
int cameraInit(VideoCapture& cap){

    // Create a capture object for the Ximea camera connected.
    int open = cap.open(CV_CAP_XIAPI);
    if(!open){
        cout << "ERROR : Ximea camera can not be opened." << endl;
        return 0;
    }

    cap.set(CV_CAP_PROP_XI_GAIN, ETH_GAIN);						// Gain Configuration.
    cap.set(CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE, ETH_SENSOR_MODE);			// Sensor Mode.
    cap.set(CV_CAP_PROP_XI_DATA_FORMAT, ETH_DATA_FORMAT);				// Data Format.
    cap.set(CV_CAP_PROP_XI_AEAG, ETH_AUTOMATIC_EXPOSURE_AND_GAIN);			// Enable/Disable AEAG.
    cap.set(CV_CAP_PROP_XI_EXPOSURE, ETH_EXPOSURE_TIME);				// Exposure Time (microseconds).
    cap.set(CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH, ETH_OUTPUT_DATA_BIT_DEPTH);       	// Pixel Size (bits).
    cap.set(CV_CAP_PROP_XI_AUTO_WB, ETH_AUTOMATIC_WHITE_BALANCE);			// Enable/Disable AWB
    cap.set(CV_CAP_PROP_XI_GAMMAY, ETH_GAMMAY);						// Gamma Y.
    cap.set(CV_CAP_PROP_XI_FRAMERATE, ETH_FRAMERATE);					// Framerate.

    // creates thread for capturing images from cam
    pthread_create(&thread_cam, NULL, getImage, (void *)& cap);

    return 1;
}

/* recordInit
 * Function creating thread for the record of the jetson camera
 */
void recordInit(){
    pthread_create(&thread_record, NULL, recordCam,  NULL);
}

/* createSocket
 * function creating a socket to send the position values via UDP to the hostmachine
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

    pthread_create(&thread_udp, nullptr, udpReceiverThread, nullptr);

    return 0;
}

/*************************************************************    Main function       ********************************************************/

/* main
 * there are 3 modes in the main function, each mode is called by writing it as the second argument when executing the program
 * - "test" mode : mode to get the HSV ranges for the color detection, with a trackbar, the clickable image to get the pixel HSV value and the thesholded mask, used for color calibration
 * - "led" mode : mode for the calibration part, using the led. image thresholded on intensity,  not used for ISC setup because of glass pannels reflecting led light
 * - "color" mode : mode taking as input the number of colors to detect and the HSV ranges and outputs the coordinates of each color detected.
 * - "colorecord" mode : same as "color" mode but saves jetson camera images also (sent to server) in a video.
 * - "colorecordtest" mode : same as "colorecord" mode but the mask is also saved in video
 * - "record" mode : only records jetson cameras (no position tracking). Used for neural network image capture
 */
int main(int argc, char *argv[])
{

    // get the chosen mode, it's the second argument when calling the program
    string mode = string(argv[1]);
    // variables for the udp socket
    char* hostMachine = argv[2];    // the third variable is the hostmachine name to send the udp messages
    char* port = argv[3];           // the fourth variable is the port
    char message[50];

    long int start, end, time_spent;
    char hostname[50];
    gethostname(hostname, HOST_NAME_MAX);
    idCam = hostname;
    transform(idCam.begin(), idCam.end(), idCam.begin(), ::toupper);

    if(mode == "record")
    {
        // initialize camera settings
        int ret = cameraInit(cap);
        if(!ret)
	{
	    cout << "ERROR : Camera Init failed." <<endl; 
            return 0;
	}

        // create socket
        createSocket(hostMachine, port);

	// Initialize the recording
        recordInit();

	// Wait for the first frame to ensure the camera is on
	while(!getFrame){}

	// Store the first frame. Useful to avoid any error if the CV Video Writer tries to save an image before a new one is acquired.
        getImageRGB();
        gpu_imageRGB.download(imageRGB);

        // Send a message to inform the system is ready.
	strcpy(message, READY_MESSAGE);
        sendto(sockfd, &message, sizeof(message), 0, (struct sockaddr*)&pcAddr, sizeof(pcAddr));

        cout<<"Ready message sent. Waiting for the start signal."<<endl;
        
        while(! okMsg){}	// TODO : Add a timeout to end the process.

        cout<<"Start message received."<<endl;
	// Compute the absolute start time
    	start = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();
	startRecord = true;

        while(!stopRecording)
	{
	    // If a frame is captured
            if(getFrame)
	    {
                // Consume the flag.
                getFrame.store(false);
		// Store the incoming frame, to be used by the record thread.
                getImageRGB();
	    }
	    else
	    {
		// Sleep to free the CPU. TODO: time slept could be a function of the framerate.
	        usleep(1000);
	    }
	}
    }

    else{
        cout<<"No mode chosen."<<endl;
        return 0;
    }


    // Compute the absolute end time
    end = duration_cast< milliseconds >( system_clock::now().time_since_epoch() ).count();

    // Wait for the threads to be closed
    // the variable "stopRecording" closes those threads
    pthread_join(thread_cam, NULL);
    pthread_join(thread_record, NULL);
    pthread_join(thread_udp, NULL);

    // Release the camera token.
    cap.release();

    // Compute the recording time.
    time_spent = end - start;
    cout << "Recording time : " << time_spent << " milliseconds" << endl;

    return 0;
}




