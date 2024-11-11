#include "resnet34.h"
#include "utils.h"
#include "imageProcess.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>


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
    if(!checkEngine())
    {
        std::cout << "start build engine file!" << std::endl;
        auto builder = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger)); // 创建构建器实例
        if (!builder)
        {
            std::cerr << "Failed to create build" << std::endl;
            return false;
        }
        uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag)); // 创建网络定义
        if (!network)
        {
            std::cerr << "Failed to create network" << std::endl;
            return false;
        }
        auto config = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());             // 创建一个配置，以此控制TensoRT如何优化网络
        auto parser = make_unique<nvonnxparser::IParser>(nvonnxparser ::createParser(*network, logger)); // 创建ONNX解析器
        if (!parser)
        {
            std::cerr << "Failed to create parser" << std::endl;
            return false;
        }
        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            std::cerr << "Failed to construct net" << std::endl;
        }

        // 注：由于导出onnx模型的时候，选择了动态模型导出，因此需要生成profile配置文件

        auto profile = builder->createOptimizationProfile(); // 创建优化配置文件
        if (!profile)
        {
            std::cerr << "Failed to create profile" << std::endl;
            return false;
        }
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最小值
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最优值
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 28, 28)); // 设置输入的最大值
        config->addOptimizationProfile(profile);

        auto serializeModel = make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config)); // 序列化模型
        if (!serializeModel)
        {
            std::cerr << "Failed to build serialize model" << std::endl;
        }
        saveEngine(serializeModel);
        return true;
    }
    else
    {
        std::cout << "have engine file!" << std::endl;
        return true;
    }
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
void Resnet34::saveEngine(make_unique<nvinfer1::IHostMemory> &serializeModel)
{
    std::ofstream sEngine(engineFilePath, std::ios::binary);  //此处以二进制读取
    if(sEngine.good())
    {
        sEngine.write(reinterpret_cast<const char*>(serializeModel->data()),serializeModel->size());
        sEngine.close();
    }
    else
    {
        std::cerr << "Could not open output file" << std::endl;
        sEngine.close();
    }
}

// 推理模型
bool Resnet34::inferNet(std::string imagePath, int &result)
{
    auto mRuntime = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));  // 创建运行时接口
    std::vector<unsigned char> engineModel = loadEgine();                               // 加载引擎文件
    auto mEngine = make_unique<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(engineModel.data(), engineModel.size())); // 反序列化生成引擎
    if(!mEngine)
    {
        std::cerr << "Failed to desrializeCudaEngine" << std::endl;
        return false;
    }
    auto mContext = make_unique<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());    // 创建上下文对象
    if(!mContext)
    {
        std::cerr << "Failed to creat context" << std::endl;
    }

    auto input_dims = mContext->getBindingDimensions(0);
    auto output_dims = mContext->getBindingDimensions(1);

    // 获取输入输出维度
    std::cout << "input_dims:" << printDims(input_dims) << std::endl; 
    std::cout << "output_dims:" << printDims(output_dims) << std::endl;
    auto input_idx = mEngine->getBindingIndex("input");
    auto output_idx = mEngine->getBindingIndex("output");

    // 获取待处理数据
    cv::Mat image;
    bool openFlag = openImage(imagePath, image);
    if(!openFlag)
    {
        std::cerr << "Failed to get image infor" << std::endl;
        return false;
    }
    // 创建cuda流，用于内存管理
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // 初始化主机内存
    float *h_buffer = nullptr;
    // 初始化设备内存, enqueueV2的输入是一个指针数组，包含输入和输出，因此此处需要一个数组
    // d_buffer[0]表示输入数据，d_buffer[1]表示输出数据
    void* d_buffer[2];
    int inputSize = getMemorySize(input_dims);
    int outputSize = getMemorySize(output_dims);

    // 申请设备内存
    cudaMalloc((void **)&d_buffer[0], inputSize*sizeof(float*));
    cudaMalloc((void **)&d_buffer[1], outputSize*sizeof(float*));
    // 申请主机内存
    h_buffer = new float[outputSize];
    cudaMemcpyAsync(d_buffer[0], image.data, inputSize*sizeof(float*), cudaMemcpyHostToDevice, stream);
    // 加入推理队列
    mContext->enqueueV2(d_buffer, stream, nullptr);
    cudaMemcpyAsync(h_buffer, d_buffer[1], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // 阻塞流，流同步
    result = getMaxIndex(h_buffer, outputSize);

    // 释放内存
    delete[] h_buffer;
    cudaFree(d_buffer[0]);
    cudaFree(d_buffer[1]);
    cudaStreamDestroy(stream);
    return true;
}

// 加载引擎
std::vector<unsigned char> Resnet34::loadEgine()
{
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if(engineFile.good())
    {
        engineFile.seekg(0, std::ios::end);
        size_t size = engineFile.tellg(); // 获取文件指针的位置
        std::vector<unsigned char> data(size);
        engineFile.seekg(0, std::ios::beg);
        engineFile.read((char *)data.data(), size);
        engineFile.close();
        return data;
    }
    else
    {
        std::vector<unsigned char> data = {};
        std::cerr << "Failed to open engineFile" << std::endl;
        engineFile.close();
        return data;
    }
}

// 检查资源目录是否有引擎文件
bool Resnet34::checkEngine()
{
    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if(engineFile.good())
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 获取最大值的下标
int Resnet34::getMaxIndex(float *h_buffer, const int len)
{
    int maxIndex;
    float maxValue = -1;
    for (int i = 0; i < len; i++)
    {
        h_buffer[i] > maxValue ? maxValue = h_buffer[i], maxIndex = i : false;
    }
    return maxIndex;
}
