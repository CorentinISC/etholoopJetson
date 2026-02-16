// White balance, fixed so the color filtering is always the same
#define WB_BLUE 2
#define WB_GREEN 1
#define WB_RED 1

#define OUTPUT_DIRECTORY (string)("/home/nvidia/Desktop/JetsonData/")

// See details at ximea.com
#define ETH_GAIN (float)(0.0)	// camera gain
#define ETH_SENSOR_MODE (int)(0)	// put camera on Zero ROT mode.
#define ETH_DATA_FORMAT (int)(5)	// set capturing mode for RAW 8
#define ETH_AUTOMATIC_EXPOSURE_AND_GAIN (int)(0)	// no automatic adjusment of exposure and gain
#define ETH_EXPOSURE_TIME	(int)(5000)	// set exposure (value in microseconds)
#define ETH_OUTPUT_DATA_BIT_DEPTH	(int)(8)	// pixel size = 8 bits
#define ETH_AUTOMATIC_WHITE_BALANCE (int)(0)	// no auto white background configurations
#define ETH_GAMMAY (float)(0.3)
#define ETH_FRAMERATE (int)(60)			// Ximea Framerate. Usually set as TARGET_FPS, but could be higher for downsampling purpose.

#define BROADCAST_PORT 5000
const char* READY_MESSAGE = "GO";  
const char* START_MESSAGE = "OK";
const char* STOP_MESSAGE = "STOP";

const int TARGET_FPS = 60;			// Video framerate.
const int64_t FRAME_PERIOD_MS = (int64_t)(1e3 / TARGET_FPS);
const size_t MAX_QUEUE_SIZE = 30;
const long BITRATE = 3000000;
const int IFRAMEINTERVAL = 30;			// Interval of frames during which images are not fully saved, only differences with the first one.
