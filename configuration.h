#define ILOWV 20

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
#define ETH_EXPOSURE_TIME	(int)(12000)	// set exposure (value in microseconds)
#define ETH_OUTPUT_DATA_BIT_DEPTH	(int)(8)	// pixel size = 8 bits
#define ETH_AUTOMATIC_WHITE_BALANCE (int)(0)	// no auto white background configurations
#define ETH_GAMMAY (float)(0.3)
#define ETH_FRAMERATE (int)(60)

#define BROADCAST_PORT 5000
const char* READY_MESSAGE = "GO";  
const char* START_MESSAGE = "OK";
const char* STOP_MESSAGE = "STOP";
