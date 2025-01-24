// resnet34模型类
#include <string>
#include <memory>
#include <vector>
#include "infer_utils.h"
#include "imageProcess.h"

/**
 * \brief Resnet34网络推理类
 */
class Resnet34 : public InferUtils
{
    private:
        std::string m_onnx_file;    // onnx文件位置
        std::string m_engine_file;  // 引擎文件保存地址

    public:
        Resnet34(const InferParams &p);
        ~Resnet34(){};

        /**
         * \brief 运行推理Resnet34分类网络模型
         * \param image_path 图片路径
         * \param sort_result 输出结果
         */
        void Run(std::string image_path, int &sort_result);


};

