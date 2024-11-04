# 将pytorch模型转化为onnx格式
import torch

onnx_file_name = "./models/resnet34.onnx"
model = torch.load("./models/resnet34.pkl")
model.eval()
batch_size = 1
dummy_input = torch.randn(batch_size, 1, 28, 28, requires_grad=True)

# 导出模型
torch.onnx.export(model,  # 模型的名称
                  dummy_input,  # 一组实例化输入
                  onnx_file_name,  # 文件保存路径/名称
                  export_params=True,  # 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                  opset_version=10,  # ONNX 算子集的版本，当前已更新到15
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],  # 输入模型的张量的名称
                  output_names=['output'],  # 输出模型的张量的名称
                  # dynamic_axes将batch_size的维度指定为动态，
                  # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})
