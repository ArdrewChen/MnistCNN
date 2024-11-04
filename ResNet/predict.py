from PIL import Image
from torchvision import transforms
import torch

transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081)
])


def predict(img_path, model_path):
    # 读取图片
    img = Image.open(img_path)

    # 图片预处理满足模型需求
    # png图片是四通道转化为3通道
    img = img.convert("RGB")
    # 转为灰度图
    img = img.convert("L")
    # 二值化
    img = img.point(lambda x: 255 if x >= 100 else 0)
    # img.show()
    # 由于训练数据以黑色为底的图片，预测图片为白色为底，需要进行转换
    img = img.point(lambda x: 255 - x)
    # img.show()
    # 图片尺寸修改，转化为张量
    img = transforms(img)
    # 加载模型
    my_model = torch.load(model_path)

    # 预测
    img = torch.reshape(img, (1, 1, 28, 28))
    my_model.eval()
    with torch.no_grad():
        result = my_model(img)
        # print(result)
        result = result.argmax(1)
        print(result.item())
        return result


if __name__ == "__main__":
    # 预测数据
    for i in range(6):
        img_path = "../temp/" + str(i) + ".png"
        model_path = "./models/resnet34.pkl"
        predict(img_path, model_path)
