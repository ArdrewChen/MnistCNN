cmake_minimum_required(VERSION 3.15)
project(MoudleDeploy VERSION 1.0 LANGUAGES CXX CUDA)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

# 设置CUDA的安装路径
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
# 设置TensorRT的安装路径
set(TENSORRT_ROOT /data/Evo/TensorRT/TensorRT-8.6.1.6)
# 设置OpenCV路径
set(OpenCV_DIR /data/Evo/opencv/build/)
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 添加CUDA头文件路径
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

# 添加TensorRT的include目录
include_directories(${TENSORRT_ROOT}/include)

# 添加TensorRT的库目录
link_directories(${TENSORRT_ROOT}/lib)

# 显式指定库路径，查找TensorRT库
find_library(TENSORRT_LIB_NVINFER NAMES nvinfer PATHS ${TENSORRT_ROOT}/lib)
find_library(TENSORRT_LIB_NVINFER_PLUGIN NAMES nvinfer_plugin PATHS ${TENSORRT_ROOT}/lib)

if (NOT TENSORRT_LIB_NVINFER OR NOT TENSORRT_LIB_NVINFER_PLUGIN)
    message(FATAL_ERROR "TensorRT libraries nvinfer or nvinfer_plugin not found!")
endif()

# 添加编译选项 -Wno-deprecated-declarations
add_compile_options(-Wno-deprecated-declarations)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC ) 
add_executable(resnet ${SRC})

# 链接OpenCV库
target_link_libraries(resnet ${OpenCV_LIBS})

# 链接TensorRT库
target_link_libraries(resnet ${TENSORRT_LIB_NVINFER} ${TENSORRT_LIB_NVINFER_PLUGIN})
target_link_libraries(resnet nvonnxparser)
# 设置环境变量，确保运行时可以找到TensorRT库
set(ENV{PATH} "/data/Evo/TensorRT/TensorRT-8.6.1.6/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "/data/Evo/TensorRT/TensorRT-8.6.1.6/lib:$ENV{LD_LIBRARY_PATH}")