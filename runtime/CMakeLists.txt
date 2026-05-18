cmake_minimum_required(VERSION 3.10)
message(STATUS "cmake version ${CMAKE_VERSION}")
project(rknn_yolov5_cam)

set(TOOLCHAIN_DIR /opt/atk-dlrk3588-toolchain)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-gcc)
set(SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot/usr/include)
set(CMAKE_SYSROOT ${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)

set(rknpu_yolov5_file rknpu2/yolov5.cc)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

include_directories(${SYSROOT})
include_directories(${SYSROOT}/opencv4)
set(OPENCV_LIBS opencv_core opencv_flann opencv_videoio opencv_video opencv_highgui opencv_imgcodecs opencv_imgproc opencv_dnn)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(RUNTIME_SRC
    app/main.cc
    engine/pipeline.cpp
    engine/thread_pool.cpp
    platform/model_coordinator.cpp
    io/camera_source.cpp
    io/frame_transform.cpp
    viz/opencv_display_sink.cpp
    viz/result_overlay.cpp
    adapters/yolo/yolo_adapter.cpp
    adapters/yolo/yolo_postprocess.cpp
    adapters/scrfd/scrfd_adapter.cpp
    adapters/scrfd/scrfd_postprocess.cpp
    adapters/ppocr/ppocr_adapter.cpp
    adapters/ppocr/ppocr_system.cc
    adapters/ppocr/postprocess.cc
    adapters/ppocr/clipper.cc
)

add_executable(edgeai_platform_demo ${RUNTIME_SRC})

target_link_libraries(edgeai_platform_demo
    imageutils
    fileutils
    imagedrawing
    ${OPENCV_LIBS}
    ${LIBRKNNRT}
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(edgeai_platform_demo Threads::Threads)

target_include_directories(edgeai_platform_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/yolo
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/scrfd
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/ppocr
    ${LIBRKNNRT_INCLUDES}
)

install(TARGETS edgeai_platform_demo DESTINATION .)
install(FILES ${LIBRKNNRT} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION ./model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_SOURCE_DIR}/config/*.yaml")
install(FILES ${CONFIG_FILES} DESTINATION config)
