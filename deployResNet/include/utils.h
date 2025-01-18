#include <string>
#include "NvInfer.h"
#include "NvOnnxParser.h"

/**
 * \brief 打印Dims格式数据的维度大小值，返回维度大小的字符串
 * \param 输入的dims数据
 **/
std::string printDims(const nvinfer1::Dims dims);

/** 
 * \brief 获取dims数据转化为一维数组的大小，用于申请cuda内存，输出总大小
 * \param 需要转化的dims数据
 **/
int getMemorySize(const nvinfer1::Dims dims);

/**
 * \brief 错误处理函数，当发生错误时，打印错误信息，并退出
 * \param condition 错误成立条件
 * \param errmasg 错误信息
 */
void errif(bool condition, const char *errmsg);