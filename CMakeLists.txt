cmake_minimum_required(VERSION 2.8)
project(jetson_proc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# --- GStreamer ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP REQUIRED gstreamer-app-1.0)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${GSTAPP_INCLUDE_DIRS}
)

link_directories(
    ${GST_LIBRARY_DIRS}
    ${GSTAPP_LIBRARY_DIRS}
)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

cuda_add_executable(jetson_proc video.cu)

target_link_libraries(jetson_proc
    -lpthread
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${GST_LIBRARIES}
    ${GSTAPP_LIBRARIES}
)
