# 开发日期：2023/9/1
import numpy as np

def identity_function(x):
    return x

def step_function(x):
    return np.int32(x>0)  #或者a=x>0   a.astype(int)

def sigmoid1(x):
    x1=x.flatten()     #先将x展开
    y=np.zeros_like(x1,dtype=float)   #这里要将整数型数组转为浮点型的数组
    for i in range(len(x1)):
        if x1[i]<0:     #分开考虑是为了防止溢出
            y[i]=np.exp(x1[i])/1+np.exp(x1[i])   #一定一定要注意首先要搞清楚数组的类型，浮点型数据输入整数型数组时会被截断
        else:
            y[i]=1/(1+np.exp(-x1[i]))
    return y.reshape(x.shape)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    if x.ndim==2:
        x=x.T
        x-=np.sum(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    c=np.max(x)
    return np.exp(x-c)/np.sum(np.exp(x-c))

def cross_entropy_error(y,t):
    if np.ndim(y) == 1:  # 这里是为了当y是一维时，让y.shape[0]=1
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    if t.size == y.size:
        t = t.argmax(axis=1)  # 将正确的标签索引记录在t中
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 一定不要忘了这里要除以batch_size



