# 检验onnx模型文件是否可用
import onnx

# 我们可以使用异常处理的方法进行检验
try:
    # 当我们的模型不可用时，将会报出异常
    onnx.checker.check_model("./models/resnet34.onnx")
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s" % e)
else:
    # 模型可用时，将不会报出异常，并会输出“The model is valid!”
    print("The model is valid!")
