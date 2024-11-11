# include "imageProcess.h"

bool openImage(std::string imagePath, cv::Mat &imageRes)
{
    cv::Mat img = cv::imread(imagePath);
    if(img.empty())
    {
        std::cerr << "Failed to open image" << std::endl;
        return false;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img ,img, {28,28});
    img = ~img;             // 白底黑字转化为黑底白字
    img.convertTo(img, CV_32F);  // 转化为浮点型
    img /= 255;         // 归一化
    hwc2chw(img);
    imageRes = img;
    
    // 打印图片信息
    // std::cout << "宽度： " << img.cols << std::endl;
    // std::cout << "高度： " << img.rows << std::endl;
    // std::cout << "通道数： " << img.channels() << std::endl;
    // // elemSize函数返回的是一个像素占用的字节数
    // std::cout << "深度： " << img.elemSize() / img.channels() * 8 << std::endl;
    return true;
}


// opencv中存储图片的格式为(h,w,c), pytorch使用的是(c,h,w)，需要手动转化
void hwc2chw(cv::Mat &img)
{
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    // 尺寸转化为(h*w, c, 1)
    img = img.reshape(1,h*w);
    // 图像转置(c, h*w, 1)
    img = img.t();
    // 转化为(c, h, w)
    img = img.reshape(w,c);
}