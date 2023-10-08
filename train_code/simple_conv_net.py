# 开发日期：2023/9/14
# CNN搭建
# 搭建的结构为“Convolution—Relu—Pooling—Affine—Relu—Affine—Softmax”
import numpy as np
import os, sys
import pickle
sys.path.append(os.pardir)

from layers import *
from collections import OrderedDict


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},  # 这里面是卷积层中的超参数
                 hidden_szie=100, output_size=10, weight_int_std=0.01):  # 括号里面都是我们要确定的超参数
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        conv_output_size = int((input_dim[1] - filter_size + 2 * pad) / stride + 1)  # 这是默认了长和宽相等,因为输入数据的长和宽是28*28
        # 注意这里的池化的高和宽以及步幅都默认是2，并且pool_output_size得到的是池化并展开成一列后的大小
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 权重和偏置的初始化
        self.param = {}
        self.param['W1'] = weight_int_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.param['b1'] = np.zeros(filter_size)
        self.param['W2'] = weight_int_std * np.random.randn(pool_output_size, hidden_szie)
        self.param['b2'] = np.zeros(hidden_szie)
        self.param['W3'] = weight_int_std * np.random.randn(hidden_szie, output_size)
        self.param['b3'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Conv'] = Convolution(self.param['W1'], self.param['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # 注意卷积层和池化层中的步幅和填充是不一样的
        self.layers['Affine1'] = Affine(self.param['W2'], self.param['b2'])  #注意这里的Affine层和之前有些不同，它的输入可能是（三维或四维）张量也可能是（一维或二维）矩阵，但输出是矩阵。
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.param['W3'], self.param['b3'])
        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastlayer.forward(y, t)
        return loss

    def accuracy(self, x, t):  # 注意你这里x是一个四维的数据
        y = self.predict(x)
        y_max = np.argmax(y, axis=1)
        if np.ndim(t) == 2:
            t_max = np.argmax(t, axis=1)
        return np.sum(y_max=t_max) / float(len(t))

    def accuracy2(self, x, t, batch_size=100):  # 为什么进一步分成细小的批量来处理
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = self.lastlayer.backward(dout=1)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 将所有参数都放入grads字典里
        grads = {}
        grads['W1'] = self.layers['Conv'].dW
        grads['b1'] = self.layers['Conv'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.param.items():
            params[key] = val
        with open(file_name, 'wb') as f:  # with open() as 以自动关闭文件的方式打开文件
            pickle.dumps(params, f)

    def load_params(self, file_name='params.pkl'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)  # 先把文件中的参数保存到parmas字典中
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv', 'Affine1', 'Affine2']):
       # 注意这里为什么不改变self.params[key]中的值呢？因为赋值操作会改变它的地址，而self.layers[key].W中的地址不变
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].W = self.params['b' + str(i + 1)]
