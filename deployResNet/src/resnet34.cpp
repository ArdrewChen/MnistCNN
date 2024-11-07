#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "resnet34.h"
#include <iostream>
#include <fstream>



class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 捕获所有警告消息，但忽略信息性消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

Resnet34::Resnet34(const mParams &p)
{
    modelFile = p.onnxFile;
    engineFilePath = p.engineFile;
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
        std::cerr << "Failed to create build" << std::endl;
        return false;
    }
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag)); // 创建网络定义
    if(!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }
    auto config = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());    // 创建一个配置，以此控制TensoRT如何优化网络
    auto parser = make_unique<nvonnxparser::IParser>(nvonnxparser ::createParser(*network, logger)); // 创建ONNX解析器
    if(!parser)
    {
        std::cerr << "Failed to create parser" << std::endl;
        return false;
    }
    auto constructed = constructNetwork(builder, network, config, parser);
    if(!constructed)
    {
        std::cerr << "Failed to construct net" << std::endl;
    }

// 注：由于导出onnx模型的时候，选择了动态模型导出，因此需要生成profile配置文件

    auto profile = builder->createOptimizationProfile(); //创建优化配置文件
    if(!profile)
    {
        std::cerr << "Failed to create profile" << std::endl;
        return false;
    }
    profile->setDimensions("input",nvinfer1:: OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最小值
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最优值
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最大值
    config->addOptimizationProfile(profile);

    auto serializeModel = make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!serializeModel)
    {
        std::cerr << "Failed to build serialize model" << std::endl;
    }
    saveEngine(engineFilePath, serializeModel);
    return true;
}

// 使用 ONNX 解析器创建 Onnx 网络并标记输出层
bool Resnet34::constructNetwork(make_unique<nvinfer1::IBuilder> &builder, 
make_unique<nvinfer1::INetworkDefinition> &network,make_unique<nvinfer1::IBuilderConfig>& config,
make_unique<nvonnxparser::IParser>& parser)
{
   auto parsed = parser->parseFromFile(modelFile.c_str(), static_cast<int>(Logger::Severity::kWARNING));  //从文件中解析ONNX
   if(!parsed)
   {
       std::cerr << "The onnxmodel is not supported" << std::endl;
   }
   config->setMaxWorkspaceSize(16 * (1 << 20)); // 设置最大工作空间
   /**************************************************************
    * Todo：设置推理精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    *
    ***********************************/

   return true;
}

// 保存引擎文件
void Resnet34::saveEngine(std::string engineFilePath, make_unique<nvinfer1::IHostMemory> &serializeModel)
{
    std::ofstream sEngine(engineFilePath, std::ios::binary);  //此处以二进制读取
    if(sEngine.good())
    {
        sEngine.write(reinterpret_cast<const char*>(serializeModel->data()),serializeModel->size());
    }
    else
    {
        std::cerr << "Could not open output file" << std::endl;
    }
}

// 推理模型
