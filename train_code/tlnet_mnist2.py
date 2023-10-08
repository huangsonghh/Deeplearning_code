# 开发日期：2023/9/4
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from two_layer_net2 import TwoLayerNet
from Deeplearning_code.dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

#超参数
item_num=10000  #迭代次数
batch_size=100  #一批次的数量
train_size=x_train.shape[0]  #训练数据集的样本数
learning_rate=0.1  #学习率

train_loss_list=[]   #训练数据的损失函数值组成的列表
train_acc_list=[]    #训练数据的正确率组成的列表
test_acc_list=[]        #测试数据的正确率组成的列表

epoch=max(train_size/batch_size,1)   #遍历一次数据就是一个epoch,也就是说epoch的数值等于要多少批次才能把所有数据遍历完

net = TwoLayerNet(input_size=784,hid_size=50,output_size=10)

for i in range(item_num):
    batch_mask=np.random.choice(train_size,batch_size) #从0到train_size-1中随机选取batch_size个数，记录数值并返回
    x_mask=x_train[batch_mask]
    t_mask=t_train[batch_mask]

    #计算该批次参数的梯度
    #grads=net.numerical_gradient(x_mask,t_mask)   #这个要运行很久很久。。。。
    #grads=net.gradient(x_mask,t_mask)     #也可以用文件two_layer_net2中的类TwoLayerNet
    grads=net.gradient(x_mask,t_mask)     #这个是加速版

    #更新参数
    for j in ['W1','b1','W2','b2']:
        net.params[j]-=learning_rate*grads[j]   #注意这里net.params[j]是一个数组对象，是一个可变类型，因此-=后不会创建新对象

    #记录每批次的损失函数值
    loss=net.loss(x_mask,t_mask)
    train_loss_list.append(loss)  #注意用法
    #print(f'loss:{loss}')

    #每一epoch，就把所有训练数据和测试数据的正确率记录下来
    if i % epoch==0:
        train_acc_val=net.accuracy(x_train,t_train)
        test_acc_val=net.accuracy(x_test, t_test)
        train_acc_list.append(train_acc_val)
        test_acc_list.append(test_acc_val)
        print('train_accuracy:'+str(train_acc_val)+'|'+'test_accuracy:'+str(test_acc_val))

#绘制图像
x=range(len(train_acc_list))
y1=train_acc_list
y2=test_acc_list
plt.plot(x,y1,label='train_acc')
plt.plot(x,y2,label='test_acc',linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1.)
plt.legend(loc='best')
plt.show()   #不要忘了show后面有括号




