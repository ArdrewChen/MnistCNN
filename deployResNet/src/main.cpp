#include <iostream>
#include "tools.h"
#include "resnet34.h"

int main(){
    mParams parmas;
    parmas.onnxFile = "../resources/resnet34.onnx";
    parmas.engineFile = "../resources/resnet34.engine";
    Resnet34 resnet34(parmas);
    std::cout << "start build" << std::endl;

    resnet34.buildNet();
}