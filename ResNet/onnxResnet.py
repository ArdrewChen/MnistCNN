# 使用ONNX Runtime推理模型
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.1307, std=0.3081)
])

# 将tensor转为numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def onnx_runtime(img_path, model_path):
    # 处理输入数据
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = img.convert("L")
    img = img.point(lambda x: 255 if x >= 100 else 0)
    img = img.point(lambda x: 255 - x)
    input_img = transforms(img)
    input_img.unsqueeze_(0)     # 三维张量插入一个维度，变成4维张量

    # 获取ONNX Runtime推理器
    ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider',
                                                              'CPUExecutionProvider'])
    ort_input = {ort_session.get_inputs()[0].name: to_numpy(input_img)}
    # 模型推理函数，参数1:输出张量列表，参数2：输入值字典。输入张量和输出张量的名称需要和torch.onnx.export中设置的输入输出名对应
    ort_output = ort_session.run(['output'], ort_input)[0]
    result = ort_output.argmax(1)
    print(result.item())


if __name__ == "__main__":
    # 预测数据
    for i in range(6):
        img_path = "../temp/" + str(i) + ".png"
        model_path = "./models/resnet34.onnx"
        onnx_runtime(img_path, model_path)
    print("Finish!")
