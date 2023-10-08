# 开发日期：2023/9/3
# 数值法+误差反向传播法求梯度的两层神经网络实现
import os, sys
import numpy as np

sys.path.append(os.pardir)
from com_function import *   #引入com_function模块中的所有函数
from gradient import *


class TwoLayerNet:
    # 权重和偏置初始化
    def __init__(self, input_size, hid_size, output_size):
        weight_step = 0.01   #控制初始权重的方差
        self.params = {}
        self.params['W1'] = weight_step * np.random.randn(input_size, hid_size)
        self.params['b1'] = np.zeros(hid_size)
        self.params['W2'] = weight_step * np.random.randn(hid_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 向前传播
    def predict(self, x):
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        y1 = np.dot(x, w1) + b1
        z1 = sigmoid(y1)
        y2 = np.dot(z1, w2) + b2
        z2 = softmax(y2)
        return z2

    # 定义损失函数
    def loss(self, x, t):   #这里的情况你要充分考虑清楚，x和t都可能为向量或者矩阵
        y = self.predict(x)
        loss=cross_entropy_error(y,t)   #交叉熵误差
        return loss

    # 计算正确率
    def accuracy(self, x, t):  # 注意这个地方的 t可以是one-hot形式也可以不是
        y = self.predict(x)
        y_m = np.argmax(y, axis=1)
        if np.ndim(t)==2:
            t_m = np.argmax(t, axis=1)
        return np.sum(y_m == t_m) / len(x)

    # 计算各参数的梯度
    # 这个是用的数值法来求的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = gradient2(loss_W, self.params['W1'])
        grads['b1'] = gradient2(loss_W, self.params['b1'])
        grads['W2'] = gradient2(loss_W, self.params['W2'])
        grads['b2'] = gradient2(loss_W, self.params['b2'])
        return grads

    #误差反向传播法来求的梯度
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

# x=np.random.randn(100,784)
# t=np.random.rand(100,10)
# net=TwoLayerNet(784,100,10)
# print(net.params['W1'].shape)
# print(net.loss(x,t))
# a=net.numerical_gradient(x,t)
# print(a['W1'].shape)