#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "resnet34.h"
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

Resnet34::Resnet34(std::string modelFile, std::string inputName, std::string outputName):
        modelFile(modelFile), inputName(inputName), outputName(outputName)
{
    std::cout << modelFile << std::endl;
}
void Resnet34::onnxNetInit()
{

}


// 构建网络
bool Resnet34::buildNet()
{
    auto builder = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger)); // 创建构建器实例
    if(!builder)
    {
        std::cout << "Failed to create build" << std::endl;
        return false;
    }
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag)); // 创建网络定义
    if(!network)
    {
        std::cout << "Failed to create network" << std::endl;
        return false;
    }
    auto config = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());    // 创建一个配置，以此控制TensoRT如何优化网络
    auto parser = make_unique<nvonnxparser::IParser>(nvonnxparser ::createParser(*network, logger)); // 创建ONNX解析器
    if(!parser)
    {
        std::cout << "Failed to create parser" << std::endl;
        return false;
    }
    // auto constructed = nvinfer1::constructNetwork()
    return true;
}

// 使用 ONNX 解析器创建 Onnx 网络并标记输出层
bool Resnet34::constructNetwork(make_unique<nvinfer1::IBuilder> &builder, 
make_unique<nvinfer1::INetworkDefinition> &network,make_unique<nvinfer1::IBuilderConfig>& config,
make_unique<nvonnxparser::IParser>& parser)
{
   // auto parsed = parser->parseFromFile();  //从文件中解析ONNX
   return true;
}
