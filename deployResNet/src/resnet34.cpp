#include "resnet34.h"

Resnet34::Resnet34(const InferParams &p):InferUtils(p)
{
    m_onnx_file = p.onnx_file;
    m_engine_file = p.engine_file;
}

void Resnet34::Run(std::string image_path, int &sort_result)
{
    // 获取待处理数据
    cv::Mat image;
    float* infer_result = nullptr;
    int result_size = 10;       // 分类大小
    bool open_flag = openImage(image_path, image);
    if(!open_flag)
    {
        std::cerr << "Failed to get image infor" << std::endl;
        exit(-1);
    }
    infer_result = new float[result_size];
    bool run_flag = RunEngine((float*)image.data, infer_result);
    if(!run_flag)
    {
        std::cerr <<"Failed to run engine" << std::endl;
        exit(-1);
    }
    sort_result = getMaxIndex(infer_result, result_size);
}
