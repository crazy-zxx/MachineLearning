import random

import torch

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


def data_iter(features, labels, batch_size):
    num_samples = len(features)
    indices = list(range(num_samples))
    # 样本随机读取
    random.shuffle(indices)
    # 批量取
    for i in range(0, num_samples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_samples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """
    模型
    :param X: 特征
    :param w: 权重
    :param b: 偏置
    :return: 预测值
    """
    return torch.matmul(X, w) + b


def squared_loss(y_pred, y):
    """
    损失函数
    :param y_pred: 预测值
    :param y: 真实值
    :return: 平方损失
    """
    return (y - y_pred.reshape(y.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    优化函数，随机梯度下降
    :param params: 待优化参数
    :param lr: 学习率
    :param batch_size: 批量大小
    :return:
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def show_example(features, labels):
    # 训练数据可视化，绘制数据分布图
    plt.subplot(2, 1, 1)
    plt.plot(features[:, 0], labels, 'b.')
    plt.subplot(2, 1, 2)
    plt.plot(features[:, 1], labels, 'b.')
    plt.show()


def init_net():
    """
    初始化模型参数
    :return: 初始化模型参数
    """
    w = torch.normal(0, 0.1, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


def train(features, labels, w, b, batch_size, num_epochs):
    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter(features, labels, batch_size):
            # 计算当前批量损失
            l = loss(net(X, w, b), y)
            # 反向传播
            l.sum().backward()
            # 梯度下降
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')


if __name__ == '__main__':
    # 使用真实值构造训练数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    num_samples = 1000
    features, labels = generate_data(true_w, true_b, num_samples)
    print(features[0], labels[0])

    # 训练数据可视化，绘制数据分布图
    show_example(features, labels)

    # 初始化模型参数
    # 目的是更新这些参数，直到这些参数足够拟合我们的数据
    w, b = init_net()

    # --- 超参数设置 ---
    # 学习率
    lr = 0.03
    # 迭代次数
    num_epochs = 3
    # 批量大小
    batch_size = 10
    # 网络模型
    net = linreg
    # 损失函数
    loss = squared_loss

    # 训练
    train(features, labels, w, b, batch_size, num_epochs)
