// resnet34模型类
#include <string>
#include <vector>
#include <memory>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// 自定义删除器，调用InferDeleter(T)，实现删除T
struct InferDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        delete obj;
    }
};

// 参数结构体
struct mParams
{
    std::string onnxFile;
    std::string engineFile;
};

// unique_ptr指针模板类
template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

class Resnet34
{
    private:
        struct mParamsStruct
        {
            std::string onnxFileName;
            std::vector<std::string> inputTensorNames;
            std::vector<std::string> outputTensorNames;
        };
        std::string modelFile;
        std::string inputName;
        std::string outputName;
        std::string engineFilePath;

    public:
        Resnet34(const mParams &p);
        void onnxNetInit();     // 网络初始化
        bool buildNet();        // 构建网络
        // 使用 ONNX 解析器创建 Onnx 网络并标记输出层
        bool constructNetwork(make_unique<nvinfer1::IBuilder> &builder,
                              make_unique<nvinfer1::INetworkDefinition> &network,
                              make_unique<nvinfer1::IBuilderConfig> &config,
                              make_unique<nvonnxparser::IParser> &parser);
        void saveEngine(std::string enginePath, make_unique<nvinfer1::IHostMemory> &serializeModel);
};