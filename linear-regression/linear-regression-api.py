import torch
from torch import nn
from torch.utils import data

import matplotlib.pyplot as plt


def generate_data(w, b, num_samples):
    """
    生成训练数据
    :param w: 权重
    :param b: 偏置
    :param num_samples:样本数
    :return: (特征,标签)
    """
    # 高斯分布数据
    X = torch.normal(0, 1, (num_samples, len(w)))
    # 矩阵乘法
    y = torch.matmul(X, w) + b
    # 噪音
    y += torch.normal(0, 0.01, y.shape)

    return X, y.reshape((-1, 1))


def load_array(data_array, batch_size, is_train=True):
    """
    加载数据集，默认训练集
    :param data_array: 数据集
    :param batch_size: 批量大小
    :param is_train: 是否为训练集
    :return: 批量数据加载器
    """
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def show_example(features, labels):
    """
    训练数据可视化，绘制数据分布图
    :param features: 特征
    :param labels: 标签
    :return:
    """
    plt.subplot(2, 1, 1)
    plt.plot(features[:, 0], labels, 'b.')
    plt.subplot(2, 1, 2)
    plt.plot(features[:, 1], labels, 'b.')
    plt.show()


if __name__ == '__main__':
    # 获取训练数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    num_samples = 1000
    features, labels = generate_data(true_w, true_b, num_samples)

    show_example(features, labels)

    # 加载训练数据
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 指定损失函数，均方误差
    loss = nn.MSELoss()

    # 指定优化算法
    lr = 0.03
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    # 计算误差
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
