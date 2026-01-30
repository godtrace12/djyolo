project(yolo11ncnn)

cmake_minimum_required(VERSION 3.10)

#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.12.0-android/sdk/native/jni)
#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/include)
#
#find_package(OpenCV REQUIRED core imgproc)

#set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20250916-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/lib/cmake/ncnn)

find_package(ncnn REQUIRED)
aux_source_directory(. SRC_LIST)

add_library(yolo11ncnn SHARED yolo11.cpp yolo11ncnn.cpp yolo11_stripcv_det.cpp yolo5_hand_ges.cpp ${SRC_LIST})

include_directories(
        # ./ncnnvulkan  # 使用这种写法需要把代码引入头文件全部改为 "net.h" 而不是 "ncnn/net.h" 形式
        ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/include/  # 懒的改代码了需要用这种方式引入 "ncnn/net.h"
)

#set_target_properties(
#        PROPERTIES IMPORTED_LOCATION
#)

target_link_libraries(yolo11ncnn ncnn)
