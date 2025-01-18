// resnet34模型类
#include <string>
#include <memory>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <vector>

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

struct Images
{
    float *inputdata;
    float *outputdata;
    int inputSize;
    int outputSize;
};

// unique_ptr指针模板类
template <typename T>
using make_unique = std::unique_ptr<T, InferDeleter>;

/**
 * \brief Resnet34网络推理类
 */
class Resnet34
{
    private:
        struct mParamsStruct
        {
            std::string onnxFileName;
            std::vector<std::string> inputTensorNames;
            std::vector<std::string> outputTensorNames;
        };
        Images images;
        // 输入数据结构体
        std::string modelFile;
        std::string inputName;
        std::string outputName;
        std::string engineFilePath;
        bool checkEngine();
        int getMaxIndex(float *h_buffer, const int len);

    public : Resnet34(const mParams &p);
        void onnxNetInit();     // 网络初始化
        bool buildNet();        // 构建网络
        // 使用 ONNX 解析器创建 Onnx 网络并标记输出层
        bool constructNetwork(make_unique<nvinfer1::IBuilder> &builder,
                              make_unique<nvinfer1::INetworkDefinition> &network,
                              make_unique<nvinfer1::IBuilderConfig> &config,
                              make_unique<nvonnxparser::IParser> &parser);
        // 保存引擎文件
        void saveEngine(make_unique<nvinfer1::IHostMemory> &serializeModel);
        // 执行推理
        bool inferNet(std::string imagePath, int &result);
        std::vector<unsigned char> loadEgine();
};