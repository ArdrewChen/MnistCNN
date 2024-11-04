# 加载模型
import glob
import os
import torch
from Net import Net
from torchvision import transforms
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def LoadModelAndUse(imgPath: str):
    """ 加载已保存的模型,并验证使用 """
    # 创建一个新的模型实例
    model = Net()
    # 加载模型
    model.load_state_dict(torch.load("./models/LeNet.pkl"))

    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转为灰度图
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化

    ])
    input_image = Image.open(imgPath)
    input_tensor = transform(input_image.resize((28, 28)))
    input_tensor = input_tensor.unsqueeze(0)  # 添加一个维度，模拟批次

    # 展示图像
    #plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
    #plt.show()

    # 使用模型进行识别
    with torch.no_grad():
        outputs = model(input_tensor)

    # 获取模型的预测结果
    _, result = torch.max(outputs, 1)
    print(f"识别结果: {result.item()}")

if __name__ =="__main__":
    LoadModelAndUse("../temp/4.png")