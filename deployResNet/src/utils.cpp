// 工具类
#include "utils.h"

/**
 * 打印Dims格式数据的维度大小值，返回维度大小的字符串
 * 参数：输入的dims数据
 **/
std::string printDims(const nvinfer1::Dims dims)
{
    int n = 0;
    char buff[100];
    std::string result;

    // snprintf用于安全地将格式化字符串写入缓冲区，
    // 参数为目标缓冲区， 目标缓冲区大小， 格式化字符串， 需要写入的内容， 返回成功写入的海报字节数
    n += snprintf(buff + n, sizeof(buff) - n, "[ ");
    for (int i = 0; i < dims.nbDims; i++)
    {
        // dims.d[i], 获取对应维度的值，如1*1*28*28，输出的d[2]为28
        n += snprintf(buff + n, sizeof(buff) - n, "%d", dims.d[i]);
        if (i != dims.nbDims - 1)
        {
            n += snprintf(buff + n, sizeof(buff) - n, ", ");
        }
    }
    n += snprintf(buff + n, sizeof(buff) - n, " ]");
    result = buff;
    return result;
}

/** 
 * 获取dims数据转化为一维数组的大小，用于申请cuda内存，输出总大小
 * 参数：需要转化的dims数据
 **/
int getMemorySize(const nvinfer1::Dims dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size = size * dims.d[i];
    }
    return size;
}
