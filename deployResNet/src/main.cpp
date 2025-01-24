#include "resnet34.h"

int main(){
    int result;
    InferParams m_parmas;
    m_parmas.engine_file = "../resources/resnet34_2.engine";
    m_parmas.onnx_file = "../resources/resnet34_2.onnx";
    m_parmas.is_dynamic_input = true;
    m_parmas.min_input_dims4 = nvinfer1::Dims4(1, 1, 28, 28);
    m_parmas.opt_input_dims4 = nvinfer1::Dims4(1, 1, 28, 28);
    m_parmas.max_input_dims4 = nvinfer1::Dims4(1, 1, 28, 28);

    Resnet34 resnet34(m_parmas);
    for (int i = 1; i < 6; i++)
    {
        std::string imagePath = "../../temp/" + std::to_string(i) + ".png";
        std::cout << imagePath << std::endl;
        resnet34.Run(imagePath, result);
        std::cout << "Infer result:" <<result << std::endl;
    }
        
}