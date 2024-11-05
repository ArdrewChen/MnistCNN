#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "resnet34.h"
#include <iostream>

using namespace nvinfer1;
class Logger : public ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 捕获所有警告消息，但忽略信息性消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

    void Resnet34::onnxNetInit()
    {
        
        
    }

    // 构建网络
    bool Resnet34::buildNet()
    {
        auto builder = createInferBuilder(logger); // 创建构建器实例
        if(!builder)
        {
            return false;
        }
        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(flag);   //创建网络定义
        return true;
    }
