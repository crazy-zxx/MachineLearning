import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    test_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [test_labels[int(i)] for i in labels]


def show_image_example(imgs, num_rows, num_cols, titles=None):
    for i, img in enumerate(imgs):
        # 一共num_rows行，num_cols列个子图，当前是第i+1个子图
        plt.subplot(num_rows, num_cols, i + 1)
        # 显示数据为图片
        plt.imshow(img)
        # 隐藏坐标轴
        plt.axis('off')
        if titles:
            # 设置标题
            plt.title(titles[i], fontsize=8)
    # 自动调整子图间距
    plt.tight_layout()
    # 显示
    plt.show()


def load_data_fashion_mnist(batch_size, resize=None, num_workers=4):
    # ToTensor()把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，
    # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
    trans = [transforms.ToTensor()]
    if resize:
        trans.append(transforms.Resize(resize))
    # Compose()组合多个transforms
    trans = transforms.Compose(trans)
    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(root='./fashionMnistData', train=True, transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./fashionMnistData', train=False, transform=trans,
                                                   download=True)
    # 返回数据加载器
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def net(X, W, b):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_pred, y):
    return -torch.log(y_pred[range(len(y_pred), y)])


def accuracy(y_pred, y):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)
        cmp = y_pred.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


if __name__ == '__main__':
    train_iter, _ = load_data_fashion_mnist(18, resize=64)
    X, y = next(iter(train_iter))
    show_image_example(X.reshape(18, X.shape[2], X.shape[3]), 2, 9, titles=get_fashion_mnist_labels(y))

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(256)

    # 28*28==784个像素点
    num_inputs = 784
    # 10个类别
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
