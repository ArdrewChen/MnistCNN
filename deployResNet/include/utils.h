#include <string>
#include "NvInfer.h"
#include "NvOnnxParser.h"

/**
 * 打印Dims格式数据的维度大小值，返回维度大小的字符串
 * 参数：输入的dims数据
 **/
std::string printDims(const nvinfer1::Dims dims);

/**
 * 获取dims数据转化为一维数组的大小，用于申请cuda内存，输出总大小
 * 参数：需要转化的dims数据
 **/
int getMemorySize(const nvinfer1::Dims dims);