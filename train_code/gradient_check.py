# 开发日期：2023/9/5
import os,sys
sys.path.append(os.pardir) 
import numpy as np

from Deeplearning_code.dataset.mnist import load_mnist
from two_layer_net2 import TwoLayerNet

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

network=TwoLayerNet(input_size=784,hid_size=50,output_size=10)

x_batch=x_train[:3]
t_batch=t_train[:3]

grads1=network.gradient(x_batch,t_batch)
grads2=network.numerical_gradient(x_batch,t_batch)

#求各个权重的绝对值误差平均
for key in grads1.keys():   #这里注意是keys，而不是key
    diff=np.average(np.abs(grads2[key]-grads1[key]))
    print(key+':'+str(diff))

def add():
    pass