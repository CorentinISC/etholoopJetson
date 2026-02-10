cmake_minimum_required(VERSION 3.10)
project(YOLO_JETSON LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Trouver les packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Inclure le dossier courant pour les headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(/opt/XIMEA/include)
include_directories(/usr/include/aarch64-linux-gnu)
include_directories(/usr/local/cuda/include)

# Ajouter les sources
set(SOURCES
    main.cpp
    ximea.cpp
    frame_queue.cpp
    yolo_trt.cpp
    video_writer.cpp
)

# Créer l'exécutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Lier OpenCV et Ximea
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS} 
    m3api 
    Threads::Threads
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
)