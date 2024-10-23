# 加载模型
import glob
import os
import torch
from Net import Net as model
from torchvision import transforms
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

# 读取图片
file_origin = glob.glob(os.path.join('../source/', '*'))  # 获取 source 目录下所有文件和子目录的路径，并将这些路径以列表的形式返回。
for i, image_origin in enumerate(file_origin):  # 同时获取可迭代对象的索引和对应的元素。它返回一个枚举对象，该对象生成由索引和元素组成的元组
    src = cv.imread(image_origin)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 白底黑字转换为黑底白字
    cv.imwrite('../temp/new{}.png'.format(i + 1), binary)

file_processed = glob.glob(os.path.join('../temp/', '*'))
rows = len(file_processed) / 5 + 1

# 先实例化，再读取
model = model()
model.eval()  # 将模型设置为“评估模式"
model.load_state_dict(torch.load('./models/LeNet.pkl'))

for i, image_processed in enumerate(file_processed):
    images = Image.open(image_processed).resize((28, 28))  # 加载图片并调整图片大小
    images = images.convert('L')  # 灰度化
    transform = transforms.ToTensor()  # 转换为张量
    image_data = transform(images).reshape(1, 1, 28, 28)  # 这里需要从resize改为reshape
    output = model(image_data)
    pred = output.argmax(dim=1)  # 返回最大值的索引
    plt.subplot(4, 5, i + 1)
    plt.tight_layout()  # 调整子图间隔
    plt.imshow(images, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.title("{}".format(int(pred)), fontsize=12)
    if i > 14:
        break
plt.savefig("../images/test_output.png")
