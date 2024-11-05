// resnet34模型类
class Resnet34
{       
    public:
        Resnet34()
        {

        }
        void onnxNetInit();     // 网络初始化
        bool buildNet();        // 构建网络
};