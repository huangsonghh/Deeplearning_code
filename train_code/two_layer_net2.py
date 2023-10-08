# 开发日期：2023/9/5
# 数值法+误差反向传播法的神经网络实现
# 与two_layer_net 相比，这个有更明显的优点，可以实现任意层神经网络的生成，因为这里是通过定义“层”来实现的
import os,sys
import numpy as np
sys.path.append(os.pardir)
from layers import *
from gradient import *
from collections import OrderedDict

class TwoLayerNet:
    # 权重和偏置初始化
    def __init__(self, input_size, hid_size, output_size):
        weight_step = 0.01   #控制初始权重的方差
        self.params = {}
        self.params['W1'] = weight_step * np.random.randn(input_size, hid_size)
        self.params['b1'] = np.zeros(hid_size)
        self.params['W2'] = weight_step * np.random.randn(hid_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层（把层给定义好）
        self.layers = OrderedDict()  #定义了self.layers用于储存“层”，并且是有序的

        # 下面这样定义以后self.layers['Affine1']就成了Affine类的实例
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Sigmoid'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastlayer = SoftmaxWithLoss()       #注意这里没有把SoftWithLoss放到字典里，这是为了后面predict

    def predict(self,x):     #调用“层”里的向前传播来完成
        for layer in self.layers.values():   #遍历字典self.layers中的值
            x=layer.forward(x)
        return x

    def loss(self,x,t):
        y=self.predict(x)
        z=self.lastlayer.forward(y,t)
        return z

    def accuracy(self,x,t):
        y = self.predict(x)
        y_m = np.argmax(y, axis=1)
        if np.ndim(t) == 2:
            t_m = np.argmax(t, axis=1)
        return np.sum(y_m == t_m) / float(x.shape[0])

    # 计算各参数的梯度
    # 这个是用的数值法来求的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient3(loss_W, self.params['W1'])   #这里可以选用gradient2也可以用numerical_gradient3
        grads['b1'] = numerical_gradient3(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient3(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient3(loss_W, self.params['b2'])
        return grads

    # 误差反向传播法来求的梯度
    def gradient(self,x,t):
        #注意这里要先forward，因为backward中用到了公共变量，要先经过forward改变数值以后才是我们想要的
        #forward
        self.loss(x,t)   #从头到尾forward了一遍

        #backward
        dout=self.lastlayer.backward()
        layers = list(self.layers.values())  # 先把字典中的值弄出来生成一个列表
        layers.reverse()  # 将列表反转
        for layer in layers:
            dout = layer.backward(dout)

        #把各参数的梯度都找出来
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
