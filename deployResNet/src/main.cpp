#include <iostream>
#include "tools.h"
#include "resnet34.h"
#include "imageProcess.h"

int main(){
    mParams parmas;
    int result;
    parmas.onnxFile = "../resources/resnet34.onnx";
    parmas.engineFile = "../resources/resnet34.engine";
    Resnet34 resnet34(parmas);
    resnet34.buildNet();
    for (int i = 0; i < 6; i++)
    {
        std::string imagePath = "../../temp/" + std::to_string(i) + ".png";
        std::cout << imagePath << std::endl;
        resnet34.inferNet(imagePath, result);
        std::cout << "Infer result:" <<result << std::endl;
    }
        
}