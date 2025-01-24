#include <string>
#include "NvInfer.h"
#include "NvOnnxParser.h"

/**
 * \brief 打印Dims格式数据的维度大小值，返回维度大小的字符串
 * \param 输入的dims数据
 **/
std::string printDims(const nvinfer1::Dims dims);

/** 
 * \brief 获取dims数据转化为一维数组的大小，用于申请cuda内存，输出总大小
 * \param 需要转化的dims数据
 **/
int getMemorySize(const nvinfer1::Dims dims);

/**
 * \brief 错误处理函数，当发生错误时，打印错误信息，并退出
 * \param condition 错误成立条件
 * \param errmasg 错误信息
 */
void errif(bool condition, const char *errmsg);

/**
 * \brief 获取h_buffer数组中的最大值下标值，用于获取分类网络输出结果
 * \param h_buffer 推理输出的概率值结果数组
 * \param len 数组长度
 */
int getMaxIndex(float *h_buffer, const int len);


/**
 * \brief 网络推理设置参数结构体
 */
struct InferParams{
    std::string onnx_file;   // onnx文件位置
    std::string engine_file; // 引擎文件保存地址
    bool is_dynamic_input;   // 是否为动态输入模型
    nvinfer1::Dims4 min_input_dims4;    // 动态输入最小尺寸
    nvinfer1::Dims4 opt_input_dims4;    // 动态输入最优尺寸
    nvinfer1::Dims4 max_input_dims4;    // 动态输入最大尺寸
};