// 工具类
#include "utils.h"

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

int getMemorySize(const nvinfer1::Dims dims)
{
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++)
    {
        size = size * dims.d[i];
    }
    return size;
}

void errif(bool condition, const char *errmsg)
{
    if(condition)
    {
        perror(errmsg);
        exit(EXIT_FAILURE);
    }
}

