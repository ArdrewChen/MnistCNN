import os.path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import resnet34
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

# 超参数设置
batch_size = 1000
EPOCHS = 60
learning_rate = 0.001
momentum = 0.5
model_path = "./models/resnet34.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定参数
current_train_step = 0
acc = 0
test_count = 0
writer = None

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

model = resnet34()

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
    torch.save(model, model_path)
