# include "imageProcess.h"
#include <vector>

bool openImage(std::string imagePath, cv::Mat &imageRes)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    // 将图像数据转换为数组
    // std::vector<uchar> array;
    // array.assign(img.data, img.data + img.total() * img.channels());
    if(img.empty())
    {
        std::cerr << "Failed to open image" << std::endl;
        return false;
    }
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);  
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    cv::threshold(img, img, 100, 255, cv::THRESH_BINARY);
    img = 255 - img; // 白底黑字转化为黑底白字  
    cv::resize(img, img, cv::Size(28, 28));
    img.convertTo(img, CV_32F, 1.0/255.0);  // 转化为浮点型
    img = (img - 0.1307) / 0.3081;
    hwc2chw(img);
    imageRes = img;
    // // 打印图片信息
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

// 归一化调整均值和方差
void normalizeImage(cv::Mat& image,float mean_c, float std_c)
{
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    image = (image - mean[0]) / stddev[0];
    image = image * std_c + mean_c;
}