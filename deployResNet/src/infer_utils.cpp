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
    std::ofstream sEngine(m_engine_file, std::ios::binary);  //此处以二进制读取
    if(sEngine.good())
    {
        sEngine.write(reinterpret_cast<const char*>(serializeModel->data()),serializeModel->size());
        sEngine.close();
        exit(-1);
    }
    else
    {
        std::cerr << "Could not open output file" << std::endl;
        sEngine.close();
        exit(-1);
    }
}

std::vector<unsigned char> InferUtils::LoadEngine()
{
    std::ifstream engineFile(m_engine_file, std::ios::binary);
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