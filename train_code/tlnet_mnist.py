# 开发日期：2023/9/3
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from two_layer_net import TwoLayerNet
from Deeplearning_code.dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

#超参数
item_num=10000  #迭代次数
batch_size=100  #一批次的数量
train_size=x_train.shape[0]  #训练数据集的样本数
learning_rate=0.1  #学习率

train_loss_list=[]

net = TwoLayerNet(input_size=784,hid_size=50,output_size=10)

for i in range(item_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_mask=x_train[batch_mask]
    t_mask=t_train[batch_mask]

    #计算该批次参数的梯度
    #grads=net.numerical_gradient(x_mask,t_mask)   #这个要运行很久很久。。。。
    grads=net.gradient(x_mask,t_mask)     #这个是加速版


    #更新参数
    for j in ['W1','b1','W2','b2']:
        net.params[j]-=learning_rate*grads[j]

    #记录每批次的损失函数值
    loss=net.loss(x_mask,t_mask)
    train_loss_list.append(loss)  #注意用法
    print(f'loss:{loss}')

#绘制图像
plt.plot(range(item_num),train_loss_list)
plt.show()   #不要忘了show后面有括号




