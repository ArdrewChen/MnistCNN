import torch
from torch import nn


# 继承自torch.nn.Module类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Sequential将多个神经网络层按顺序组合在一起，构建一个简单的网络模型
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 卷积层, 输出6@ 28*28
            nn.AvgPool2d(kernel_size=2, stride=2),  # 平均汇聚层，输出6@14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 卷积层，输出16@10*10
            nn.AvgPool2d(kernel_size=2, stride=2),  # 平均汇聚层，输出16@5*5
        )
        self.flatten = nn.Sequential(
            nn.Flatten()  # 将多维输入张量展平成一维向量的模块，通常用于连接卷积层和全连接层，输出1*400的一维张量
        )
        self.linear = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Linear(84, 10)  # 全连接层，第一个参数为输入特征大小，第二个是输出特征大小
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
