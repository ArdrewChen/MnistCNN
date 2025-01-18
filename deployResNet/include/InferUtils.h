#include "NvInfer.h"
#include <iostream>
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 捕获所有警告消息，但忽略信息性消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


/**
 * \brief 网络推理的父类，该类主要负责tensorRT推理的常用函数的设置
 */
class InferUtils
{
    private:

    public:

        

};