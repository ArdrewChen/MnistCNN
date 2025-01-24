#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "utils.h"

/**
 * \brief 日志类
 */
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 捕获所有警告消息，但忽略信息性消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

/**
 * \brief 自定义删除器，调用InferDeleter(T)，实现删除T
 */
struct InferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};

/**
 * \brief unique_ptr指针模板类
 */
template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;


/**
 * \brief 网络推理的父类，该类主要负责tensorRT推理的常用函数的设置
 */
class InferUtils
{
    private:
        std::string m_onnx_file;    // onnx文件位置
        std::string m_engine_file;  // 引擎文件保存地址
        bool m_dynamic_flag;        // 是否为动态输入模型
        nvinfer1::Dims4 min_input_dims4;    // 动态输入最小尺寸
        nvinfer1::Dims4 opt_input_dims4;    // 动态输入最优尺寸
        nvinfer1::Dims4 max_input_dims4;    // 动态输入最大尺寸
        Logger logger;


        /**
         * \brief 检查文件中是否包含已构建完成的tensorRT引擎
         * \return 存在返回true，不存在返回false
         */
        bool CheckEngineExist();

        /**
         * \brief 使用 ONNX 解析器创建 Onnx 网络并标记输出层
         * \return 构建成功返回true,失败返回false
         */
        bool ConstructNetwork(make_unique<nvinfer1::IBuilderConfig>& config,
                              make_unique<nvonnxparser::IParser>& parser);

        /**
         * \brief 保存引擎文件
         * \param serializeModel 构造器产生的序列化模型
         */
        void SaveEngine(make_unique<nvinfer1::IHostMemory> &serializeModel);

        

    public:
        InferUtils(const InferParams& infer_params);
        ~InferUtils(){};

        /**
         * \brief 构建网络函数
         * \return 构建成功返回true,否则返回false
         */
        bool BuildEngine();

        /**
         * \brief 加载引擎
         * \return 加载成功返回true,否则返回false
         */
        std::vector<unsigned char> LoadEngine();

        /**
         * \brief 推理引擎
         * \param input_data 输入数据
         * \param infer_result 推理输出结果
         * \return 运行成功返回true,失败返回false
         */
        bool RunEngine(float* input_data, float* infer_result);
        

};