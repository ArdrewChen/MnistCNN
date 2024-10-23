# 训练模型

import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from Net import Net


# 展示数据集
def show_dataset():
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(train_dataset.data[i], cmap='gray', interpolation='none')
        plt.title("Labels: {}".format(train_dataset.targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("../images/train_dataset.png")


batch_size = 64
EPOCHS = 40
learning_rate = 0.01
momentum = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试集
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
show_dataset()

model = Net()
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
# 学习率：每次参数更新的步伐的大小，冲量：会让更新方向受前几次更新方向的影响，从而能够更快地穿越平缓区域并避免被局部最优困住。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # 优化器，用于随机梯度下降， lr是学习率，momentum是冲量

acc_list_test = []  # 测试准确率
for epoch in range(EPOCHS):
    running_loss = 0.0  # 这小批300的loss清零
    running_total = 0
    running_correct = 0  # 这小批300的acc清零
    # enumerate() 是 Python 内置函数，它允许在迭代可迭代对象时同时获取元素和对应的索引（即下标），0表示索引从0开始
    # 通过 enumerate 来遍历 train_loader，并为每个批次分配一个索引 batch_idx
    # data 是当前批次的数据，通常是一个元组，包含输入张量 inputs 和对应的标签 labels。
    for batch_idx, data in enumerate(train_loader, 0):
        # 训练
        inputs, target = data
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, target)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 沿着指定的维度 dim=1，即按行（每个样本的各类分数）找出最大值。它返回两个值
        # 第一个是每行的最大值（这些是实际的分数或概率）
        # 第二个是每行最大值所在的索引，即模型预测的类别（predicted）
        _, predicted = torch.max(outputs.detach(), dim=1)
        # 比较模型预测的类别 predicted 与真实标签 target 是否相等
        running_total += inputs.shape[0]
        # 统计出来的正确结果
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

    # 测试
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for test_data in test_loader:
            test_images, test_labels = data
            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.detach(),
                                          dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += test_labels.size(0)  # 张量之间的比较运算
            correct += (predicted == test_labels).sum().item()
    acc_test = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch + 1, EPOCHS, 100 * acc_test))  # 求测试的准确率，正确数/总数
    acc_list_test.append(acc_test)

fig = plt.figure()
torch.save(model.state_dict(), './models/LeNet.pkl')  # 保存模型
plt.plot(acc_list_test)
plt.xlabel('Epoch')
plt.ylabel('Accuracy On TestSet')
plt.savefig("../images/LeNet_acc.png")
