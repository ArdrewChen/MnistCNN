#include "infer_utils.h"

InferUtils::InferUtils(const InferParams& infer_params)
{
    m_onnx_file = infer_params.onnx_file;
    m_engine_file = infer_params.engine_file;
    m_dynamic_flag = infer_params.is_dynamic_input;
    min_input_dims4 = infer_params.min_input_dims4;
    opt_input_dims4 = infer_params.opt_input_dims4;
    max_input_dims4 = infer_params.max_input_dims4;
};

bool InferUtils::BuildEngine()
{
    if(!CheckEngineExist())
    {
         std::cout << "start build engine file!" << std::endl;
        auto builder = make_unique<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger)); // 创建构建器实例
        if (!builder)
        {
            std::cerr << "Failed to create builder" << std::endl;
            return false;
        }
        uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag)); // 创建网络定义
        if (!network)
        {
            std::cerr << "Failed to create network definition" << std::endl;
            return false;
        }
        auto config = make_unique<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());             // 创建一个配置，以此控制TensoRT如何优化网络
        auto parser = make_unique<nvonnxparser::IParser>(nvonnxparser ::createParser(*network, logger)); // 创建ONNX解析器
         if (!parser)
        {
            std::cerr << "Failed to create parser" << std::endl;
            return false;
        }
        auto constructed = ConstructNetwork(config, parser);
        if (!constructed)
        {
            std::cerr << "Failed to construct net" << std::endl;
            return false;
        }
        if(m_dynamic_flag)
        {
            auto profile = builder->createOptimizationProfile(); // 创建优化配置文件
            if (!profile)
            {
                std::cerr << "Failed to create profile" << std::endl;
                return false;
            }
            // 设置输入的最小值
            profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, min_input_dims4); 
            // 设置输入的最优值
            profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, opt_input_dims4);
            // 设置输入的最大值 
            profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, max_input_dims4); 
            config->addOptimizationProfile(profile);
        }
        auto serializeModel = make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config)); // 序列化模型
        if (!serializeModel)
        {
            std::cerr << "Failed to build serialize model" << std::endl;
        }
        SaveEngine(serializeModel);
        return true;
    }
    else
    {
        std::cout << "have engine file!" << std::endl;
        return true;
    }

}

bool InferUtils::RunEngine(float* input_data, float* infer_reuslt )
{
    // 创建运行时接口
    auto m_runtime = make_unique<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));  
    std::vector<unsigned char> engine_model = LoadEngine();                               // 加载引擎文件
    auto m_engine = make_unique<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_model.data(), engine_model.size())); // 反序列化生成引擎
    if(!m_engine)
    {
        std::cerr << "Failed to desrializeCudaEngine" << std::endl;
        return false;
    }
    auto m_context = make_unique<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());    // 创建上下文对象
    if(!m_context)
    {
        std::cerr << "Failed to creat context" << std::endl;
    }

    auto input_dims = m_context->getBindingDimensions(0);
    auto output_dims = m_context->getBindingDimensions(1);

    auto input_idx = m_engine->getBindingIndex("input");
    auto output_idx = m_engine->getBindingIndex("output");
    // 创建cuda流，用于内存管理
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // 初始化主机内存
    float *h_buffer = nullptr;
    // 初始化设备内存, enqueueV2的输入是一个指针数组，包含输入和输出，因此此处需要一个数组
    // d_buffer[0]表示输入数据，d_buffer[1]表示输出数据
    void* d_buffer[2];
    int input_size = getMemorySize(input_dims);
    int output_size = getMemorySize(output_dims);

    // 申请设备内存
    cudaMalloc((void **)&d_buffer[0], input_size*sizeof(float));
    cudaMalloc((void **)&d_buffer[1], output_size*sizeof(float));
    // 申请主机内存
    h_buffer = new float[output_size];
    cudaMemcpyAsync(d_buffer[0], input_data, input_size*sizeof(float), cudaMemcpyHostToDevice, stream);
    // 加入推理队列
    m_context->enqueueV2(d_buffer, stream, nullptr);
    cudaMemcpyAsync(h_buffer, d_buffer[1], output_size*sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // 阻塞流，流同步
    for(int i=0; i<output_size; i++)
    {
        infer_reuslt[i] = h_buffer[i];
    }

    // 释放内存
    delete[] h_buffer;
    cudaFree(d_buffer[0]);
    cudaFree(d_buffer[1]);
    cudaStreamDestroy(stream);
    return true;

}

bool InferUtils::CheckEngineExist()
{
    std::ifstream engine_file(m_engine_file, std::ios::binary);
    if(engine_file.good())
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool InferUtils::ConstructNetwork(make_unique<nvinfer1::IBuilderConfig>& config, make_unique<nvonnxparser::IParser>& parser)
{
   auto parsed = parser->parseFromFile(m_onnx_file.c_str(), static_cast<int>(Logger::Severity::kWARNING));  //从文件中解析ONNX
   if(!parsed)
   {
       std::cerr << "The onnxmodel is not supported" << std::endl;
       return false;
   }
   config->setMaxWorkspaceSize(16 * (1 << 20)); // 设置最大工作空间
   /**************************************************************
    * Todo：设置推理精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    *
    ***********************************/

   return true;
}

void InferUtils::SaveEngine(make_unique<nvinfer1::IHostMemory> &serializeModel)
{
    std::ofstream s_engine(m_engine_file, std::ios::binary);  //此处以二进制读取
    if(s_engine.good())
    {
        s_engine.write(reinterpret_cast<const char*>(serializeModel->data()),serializeModel->size());
        s_engine.close();
        exit(-1);
    }
    else
    {
        std::cerr << "Could not open output file" << std::endl;
        s_engine.close();
        exit(-1);
    }
}

std::vector<unsigned char> InferUtils::LoadEngine()
{
    std::ifstream engine_file(m_engine_file, std::ios::binary);
    if(engine_file.good())
    {
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg(); // 获取文件指针的位置
        std::vector<unsigned char> data(size);
        engine_file.seekg(0, std::ios::beg);
        engine_file.read((char *)data.data(), size);
        engine_file.close();
        return data;
    }
    else
    {
        std::vector<unsigned char> data = {};
        std::cerr << "Failed to open engineFile" << std::endl;
        engine_file.close();
        return data;
    }
}