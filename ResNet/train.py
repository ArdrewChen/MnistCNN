# 训练模型
import os.path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import resnet34
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

# 超参数设置
batch_size = 1000
EPOCHS = 60
learning_rate = 0.001
momentum = 0.5
model_path = "./models/resnet34_2.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定参数
current_train_step = 0
acc = 0
test_count = 0
writer = None

# 监测数据
loss_buffer = []
loss_total = []
train_acc_total = []
test_acc_total = []

# 训练集
train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
# 测试集
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))

# 加载训练集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义的时候要指定分类的数量，保证最后的输出正确
model = resnet34(num_classes=10)
print(model)

# 断点续训
if os.path.exists(model_path):
    model = torch.load(model_path)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(), learning_rate)
writer = SummaryWriter(log_dir="../logs")
# 训练
for item in range(EPOCHS):
    print("-----开始第%s轮测试-----" % str(item))
    model.train()
    for data in train_loader:
        imgs, target = data
        # 计算预测值
        predict = model(imgs)
        # 计算损失值
        loss = loss_fn(predict, target)
        # 梯度归0
        optimizer.zero_grad()
        # 反向计算梯度
        loss.backward()
        # 反向更新
        optimizer.step()
        # 输出损失值
        current_train_step += 1
        if current_train_step % 10 == 0:
            print("训练次数%s  损失值：%s" % (current_train_step, loss.item()))
            loss_total.append(loss.item())
            writer.add_scalar("train_loss", loss.item(), global_step=current_train_step)
    model.eval()


    # 测试
    total_acc = 0
    for data in test_loader:
        imgs, target = data
        with torch.no_grad():
            predict = model(imgs)
            acc = (predict.argmax(1) == target).sum()
            print("准确率：%s" % str(acc / imgs.shape[0]))
            total_acc += acc

    writer.add_scalar("test_acc", total_acc / test_count, global_step=current_train_step)
    test_acc_total.append(total_acc / test_count)
    torch.save(model, model_path)

fig = plt.figure()
plt.plot(loss_total)
plt.xlabel('Train Step')
plt.ylabel('Loss On TrainSet')
plt.savefig("../images/resnet34_loss.png")

fig2 = plt.figure()
plt.plot(test_acc_total)
plt.xlabel('Epoch')
plt.ylabel('Loss On TestSet')
plt.savefig("../images/resnet34_acc.png")
