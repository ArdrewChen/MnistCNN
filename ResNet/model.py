# 模型代码
import torch.nn as nn
import torch


# 第一种ResNet结构,层数较低的结构
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(out_channel)
        # relu激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样层
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 处理shortcut块间维度不匹配的问题
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 先将shortcut分支和主分支相加，然后再经过ReLU函数
        out += identity
        out = self.relu(out)

        return out


# 第二种残差块结构，层数较高的块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # block是块对象
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2,padding=3,
        # bias=False)

        # 适应输入数据集的维度
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.in_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 对应ResNet的四个部分
        # 第一块是通过汇聚层的输出得到输入，数据维度不需要变化
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应下采样池化，无论输入的维度是多少，输出的维度都是1*1
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，num_classes是输出特征

        # 遍历神经网络模型中的所有模块（modules），并对其中所有 nn.Conv2d 类型的卷积层进行权重初始化，使用kaiming初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # stride不为1,说明需要加1*1卷积核调整shortcut的维度
        # 输入通道和输出通道不相等，说明是第二种结构，3*3卷积核和最后一个卷积核的通道数不一样，也需要1*1卷积核
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 第一个残差块可能需要下采样，因此单独拎出来
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        # 每一个大块中的第一个残差块的输入是channel，从第二块开始，输入是channel*4 or 1(expansion),
        self.in_channel = channel * block.expansion
        # 得到剩余的残差块,循环为1~block_num-1,刚好少一层
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        # *用于解包操作，*layers会将列表layers中的每个元素解包为单独的参数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 output shape:\t', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('maxpool output shape:\t', x.shape)

        x = self.layer1(x)
        # print('layer1 output shape:\t', x.shape)
        x = self.layer2(x)
        # print('layer2 output shape:\t', x.shape)
        x = self.layer3(x)
        # print('layer3 output shape:\t', x.shape)
        x = self.layer4(x)
        # print('layer4 output shape:\t', x.shape)

        if self.include_top:
            x = self.avgpool(x)
            # print('avgpool output shape:\t', x.shape)
            x = torch.flatten(x, 1)
            # print('flatten output shape:\t', x.shape)
            x = self.fc(x)
            # print('fc output shape:\t', x.shape)

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
