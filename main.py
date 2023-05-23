import torch
import random
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 下载并加载训练数据
training_data = datasets.MNIST(
    root="./data/",     # 将数据下载到当前目录
    train=True,
    download=False,
    transform=ToTensor(),
)

# 准备测试集
test_data = datasets.MNIST(
    root="./data/",
    train=False,        # 注意我们这里设置了train=False
    download=False,
    transform=ToTensor(),
)

# 创建一个读取MNIST二进制文件的函数
def read_images(filename='./data/MNIST/raw/train-images-idx3-ubyte'):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16)) # 2051 60000 28 28
        '''
        `struct.unpack()` 是 Python `struct` 模块的一个函数，它用于将字节流转换为 Python 的数据类型。
        函数的第一个参数是一个格式字符串，定义了要解压的数据的结构。`'>IIII'` 表示要解压的数据包含四个无符号整数（'I' 表示无符号整数），并且数据的字节顺序是 big-endian（'> '表示 big-endian）。
        `f.read(16)` 读取了文件的前 16 个字节。由于我们知道 MNIST 数据集的文件格式，所以我们知道前 16 个字节包含四个 4 字节的整数：魔术数字、图像数量、行数和列数。
        `struct.unpack('>IIII', f.read(16))` 的结果是一个四元组，包含这四个整数的值。
        这个函数是对 MNIST 数据集文件格式的一种解析方式，它利用了我们对文件格式的知识。如果你尝试对不同格式的文件使用这个函数，可能会导致错误或返回无意义的结果。
        '''
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        '''
        np.fromfile(f, dtype=np.uint8): 这个操作会读取文件中剩余的所有字节，并将这些字节转换为 Numpy 数组。dtype=np.uint8 参数告诉 Numpy 这些字节应该被解释为无符号的 8 位整数（即 0-255 的整数，这正好对应于灰度图像的像素值）。
        .reshape(num, rows, cols): 这个操作会改变数组的形状，使其具有新的维度。在这个情况下，我们将一维数组变为了一个三维数组。num 是图像的数量，rows 和 cols 分别是图像的行数和列数。因此，这个操作将一维数组变为了一个形状为（图像数量，行数，列数）的三维数组。
        images = ...: 这个操作将新形成的三维数组赋值给 images 变量。
        '''
    return images


# 基本参数
dim = 1
lr = 1e-4
epochs = 100
batch_size = 1024

# 定义一个神经网络类，继承自nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 定义一个Flatten层，将每个28x28的图片数据转化为784的向量
        self.flatten = nn.Flatten()
        # 定义一个线性ReLU堆栈，包括两个线性层和ReLU激活层，以及一个输出层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 在forward函数中，首先进行Flatten操作，然后通过线性ReLU堆栈进行计算
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 数据集准备
train_dataloader = DataLoader(training_data, batch_size=1024)
test_dataloader = DataLoader(test_data, batch_size=1024)

# 初始化模型，损失函数，优化器，并将模型移动到相应设备（CPU或CUDA设备）
# 如果有可用的CUDA设备，将设备设置为'cuda'，否则设置为'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NeuralNetwork().to(device)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


# 定义训练函数
def train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 将数据移动到相应设备（CPU或CUDA设备）
        X, y = X.to(device), y.to(device)

        # 计算模型预测结果和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算已经处理的样本数量
        current = batch * len(X)

        # 打印训练进度
        if batch % 8 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"-------------------------------")


def train_frequency():
    for t in range(epochs):
        print(f"Epoch {t + 1}")
        train()
    # 保存模型参数
    torch.save(model.state_dict(), 'model.pth')

# 评估和推理的加载模型
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))
# 评估模式
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * len(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader.dataset)
    correct /= size
    print(f"\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 推理模式
def inference(images=read_images(), model=model):
    model.eval()
    # 60000是size = len(dataloader.dataset)
    r_num = random.randint(0, 60000)
    images = images[r_num]
    # 读取到的 image 是一个 NumPy 数组，形状为 [28, 28]，包含了你的图像
    # 我们需要将其转换为 PyTorch 张量，并添加额外的维度
    image_tensor = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
    # 确保模型和输入数据在同一设备上
    image_tensor = image_tensor.to(device)

    # 用模型进行预测
    with torch.no_grad():
        output = model(image_tensor)

    # 获取预测结果
    _, predicted = torch.max(output, dim=1)
    print("数字是：", predicted.item())
    plt.imshow(images)
    plt.show()


train_frequency()
# test(test_dataloader, model)
# inference()
