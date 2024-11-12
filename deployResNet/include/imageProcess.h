#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

bool openImage(std::string imagePath, cv::Mat &imageRes);
void hwc2chw(cv::Mat &img);
void normalizeImage(cv::Mat &image, float mean_c, float std_c);