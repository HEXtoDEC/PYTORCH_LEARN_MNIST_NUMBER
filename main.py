import torch
import random
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

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
    train=False,        # 注意这里设置了False
    download=False,
    transform=ToTensor(),
)

# 创建一个读取MNIST二进制文件的函数
def read_images(filename='.\\data\\MNIST\\raw\\train-images-idx3-ubyte'):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16)) # 2051 60000 28 28
        '''
        `struct.unpack()` 是 Python `struct` 模块的一个函数，它用于将字节流转换为 Python 的数据类型。
        函数的第一个参数是一个格式字符串，定义了要解压的数据的结构。`'>IIII'` 表示要解压的数据包含四个无符号整数（'I' 表示无符号整数），并且数据的字节顺序是 big-endian（'> '表示 big-endian）。
        `f.read(16)` 读取了文件的前 16 个字节。由于知道 MNIST 数据集的文件格式，所以知道前 16 个字节包含四个 4 字节的整数：魔术数字、图像数量、行数和列数。
        `struct.unpack('>IIII', f.read(16))` 的结果是一个四元组，包含这四个整数的值。
        这个函数是对 MNIST 数据集文件格式的一种解析方式，它利用了对文件格式的知识。如果你尝试对不同格式的文件使用这个函数，可能会导致错误或返回无意义的结果。
        '''
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        '''
        np.fromfile(f, dtype=np.uint8): 这个操作会读取文件中剩余的所有字节，并将这些字节转换为 Numpy 数组。dtype=np.uint8 参数告诉 Numpy 这些字节应该被解释为无符号的 8 位整数（即 0-255 的整数，这正好对应于灰度图像的像素值）。
        .reshape(num, rows, cols): 这个操作会改变数组的形状，使其具有新的维度。在这个情况下，将一维数组变为了一个三维数组。num 是图像的数量，rows 和 cols 分别是图像的行数和列数。因此，这个操作将一维数组变为了一个形状为（图像数量，行数，列数）的三维数组。
        images = ...: 这个操作将新形成的三维数组赋值给 images 变量。
        '''
    return images


# 基本参数
dim = 1         # 指定LogSoftmax函数的维度
lr = 1e-4       # 学习率，用于在训练过程中更新网络权重
epochs = 100    # 训练的周期数，每个周期会对整个数据集进行一次训练
batch_size = 5  # 批处理大小，每次训练会取出batch_size数量的数据进行训练

# 全连接线性层
# 定义一个神经网络类，继承自nn.Module，这是一个深度神经网络结构，它由多个线性层和激活层组成，通过这些层的堆叠，可以学习到输入数据中的复杂模式。
# 在训练过程中，会使用一种叫做反向传播的算法，来更新网络中的权重参数，使得网络的预测结果尽可能接近真实的标签。
# 这个过程通常需要通过优化算法（如：随机梯度下降）来完成，优化算法会根据网络的损失函数（预测结果和真实标签之间的差距）来更新网络的权重。
# 批量归一化层（BatchNorm）在神经网络中是非常重要的一部分，能够使得神经网络在训练过程中保持更好的稳定性，防止梯度消失或者梯度爆炸问题，从而加速网络训练的过程。同时，BatchNorm也具有一定的正则化效果，可以防止模型过拟合。
# ReLU（Rectified Linear Unit）激活函数在神经网络中用于引入非线性。因为线性操作无法模拟复杂的数据分布，通过ReLU等激活函数，可以引入非线性，让神经网络有能力学习并模拟更复杂的数据分布。
# 最后的输出层使用了LogSoftmax作为激活函数，这是因为希望网络的输出可以表示为各个类别的概率。Softmax函数可以将一组任意的实数转化为一组概率分布，LogSoftmax则是对Softmax的输出取对数。在许多情况下，使用LogSoftmax可以提高数值稳定性。
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         # 定义一个Flatten层，将每个28x28的图片数据转化为784的向量，https://pytorch.org/docs/stable/generated/torch.flatten.html
#         # 这一步是为了将二维的图片数据转化为一维，以便进行线性变换
#         self.flatten = nn.Flatten()
#
#         # 定义一个线性ReLU堆栈，包括两个线性层和ReLU激活层，以及一个输出层
#         # 线性层（nn.Linear）是神经网络的基础组成单元，用于实现线性变换，https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
#         # ReLU激活函数用于引入非线性，使得神经网络可以拟合更复杂的函数
#         # 批量归一化层（nn.BatchNorm1d）用于进行特征的归一化处理，可以加速网络训练，提高模型的泛化能力
#         # LogSoftmax用于输出层，因为在多分类问题中，一般希望网络的输出可以表示为各个类别的概率
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.LogSoftmax(dim)
#         )
#
#     def forward(self, x):
#         # 在forward函数中，首先进行Flatten操作，然后通过线性ReLU堆栈进行计算
#         # 这个过程就是神经网络的前向传播过程，输入的数据会按照定义的层的顺序，依次进行计算，最后输出预测结果
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# CNN卷积层
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        # 首先调用父类的构造函数，用于自己构造自己。
        super(ConvolutionalNetwork, self).__init__()

        # 首先定义一个Sequential模型，这个模型可以按顺序执行一系列的神经网络层
        # 这里定义了两个卷积层，每个卷积层后面都跟着一个BatchNorm层和ReLU激活函数，以及一个MaxPool层
        # 卷积层可以看作是一个滤波器，可以在输入图片上滑动，提取图片的局部特征
        # BatchNorm层可以加速神经网络的训练，它会对每个小批量的数据进行归一化操作，使得数据的分布更加稳定
        # ReLU激活函数可以增加神经网络的非线性，使得神经网络可以拟合更复杂的函数
        # MaxPool层可以进行下采样操作，减少数据的维度，同时保留最重要的特征
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # 输入通道数为1，输出通道数为32，卷积核大小为3x3，步长为1，填充为1
            nn.BatchNorm2d(32),                                     # 对32个通道的数据进行归一化
            nn.ReLU(),                                              # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 最大池化，池化核大小为2x2，步长为2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，填充为1
            nn.BatchNorm2d(64),                                     # 对64个通道的数据进行归一化
            nn.ReLU(),                                              # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 最大池化，池化核大小为2x2，步长为2
        )
        # 然后定义一个全连接层，用于将卷积层提取的特征进行分类
        # 全连接层可以看作是一个普通的神经网络，它将所有的输入连接到所有的输出
        # 这里有两个全连接层，第一个全连接层的输出大小为512，第二个全连接层的输出大小为10，因为假设有10个类别
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),     # 输入大小为64*7*7，输出大小为512
            nn.BatchNorm1d(512),            # 对512个特征进行归一化
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(512, 10),             # 输入大小为512，输出大小为10
            nn.LogSoftmax(dim=1)            # LogSoftmax激活函数，可以将网络的输出转化为各个类别的概率
        )

    def forward(self, x):
        # 在forward函数中，定义了模型的前向传播过程
        # 首先将输入数据送入卷积层
        x = self.conv_relu_stack(x)
        # 然后需要将数据展平（flatten），因为全连接层只能处理一维的数据
        # 这里使用view函数将数据展平，第一个维度保留不变（这个维度是批量大小），其余的维度合并为一维
        x = x.view(x.size(0), -1)
        # 然后将展平的数据送入全连接层
        x = self.fc(x)
        # 最后，返回网络的输出
        return x


# 数据集准备
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)

# 初始化模型，损失函数，优化器，并将模型移动到相应设备（CPU或CUDA设备）
# 如果有可用的CUDA设备，将设备设置为'cuda'，否则设置为'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvolutionalNetwork().to(device) # CNN卷积
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr)

# 添加钩子函数并定义一个字典来存储激活
activations = {}
def forward_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# 在开始训练之前注册钩子
for name, module in model.named_modules():
    module.register_forward_hook(forward_hook(name))

# 定义训练函数
def train():
    dataloader = train_dataloader
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # 将数据移动到相应设备（CPU或CUDA设备）
        x, y = x.to(device), y.to(device)

        # 计算模型预测结果和损失
        pred = model(x)
        loss = loss_fn(pred, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算已经处理的样本数量
        current = batch * len(x)

        # 打印训练进度
        loss = loss.item()
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train_frequency():
    writer = SummaryWriter()
    for t in range(epochs):
        print(f"-------------------------------")
        print(f"Epoch {t + 1}")
        train()
        # 在每个epoch结束时，向TensorBoard添加激活
        for name, activation in activations.items():
            writer.add_histogram(f'{name}.activation', activation, t)
            activation.clear()  # 清空激活值
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, t)
            writer.add_histogram(f'{name}.grad', weight.grad, t)
        activations.clear()  # 清空激活值
    writer.close()
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')


# 评估模式
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item() * len(x)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader.dataset)
    correct /= size
    print(f"\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 推理模式
def inference(images=read_images(), model=model):
    model.eval()
    # 60000是size = len(dataloader.dataset)
    r_num = random.randint(0, 60000-1)
    images = images[r_num]
    # 读取到的 image 是一个 NumPy 数组，形状为 [28, 28]，包含了你的图像
    # 将其转换为 PyTorch tensor，并添加额外的维度
    image_tensor = torch.from_numpy(images).float().unsqueeze(0).unsqueeze(0)
    # 确保模型和输入数据在同一设备上
    image_tensor = image_tensor.to(device)

    # 用模型进行预测
    with torch.no_grad():
        output = model(image_tensor)

    # 获取预测结果
    _, predicted = torch.max(output, dim)
    print("数字是：", predicted.item())
    plt.imshow(images)
    plt.show()

# 训练
train_frequency()
# CMD下执行 tensorboard --logdir=runs 启动面板
# 验证
model.load_state_dict(torch.load('model.pth'))
test(test_dataloader, model)
# 推理
inference()
