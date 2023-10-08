# 开发日期：2023/9/1
import os
import sys
import pickle

#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定(因为后面已经将父目录写出来了，所以这一部分可以不用）
import numpy as np
from Deeplearning_code.dataset.mnist import load_mnist

def get_date():
    _,(x_test,t_test)=load_mnist(flatten=True,normalize=True,one_hot_label=False) # 注意这里归一化了
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def predict(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    z1=sigmoid(np.dot(x,w1)+b1)
    z2=sigmoid(np.dot(z1,w2)+b2)
    y=softmax(np.dot(z2,w3)+b3)
    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    if x.ndim==2:
       for i in range(len(x)):
           c=np.max(x[i])
           x[i]=np.exp(x[i] - c)/np.sum(np.exp(x[i]-c))
           return x
    else:
        c = np.max(x)
        return np.exp(x - c) / np.sum(np.exp(x - c))  # 防止溢出

x,t=get_date() #x的大小是10000*784，t的大小是10000
network=init_network()
batch_size=100     #加入了批次（batch）
batch_num=len(x)/batch_size
right_count=0
for i in range(0,len(x),batch_size):
    y=predict(network,x[i:i+batch_size])
    right_count+=np.sum(np.int32(np.argmax(y,axis=1)==t[i:i+batch_size]))
print(right_count/len(t))