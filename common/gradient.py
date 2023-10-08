# coding: utf-8
import numpy as np

#数值法求梯度，针对一维的情况
def _numerical_gradient_1d(f, x):   #求 f 关于 x 的偏导也就是梯度
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):      #为什么要先去改变x中元素的值然后又去还原它呢？这其实是为之后求参数的偏导做准备
        tmp_val = x[idx]       #（接上面）因为改变了值以后，下一步调用f（x），也就是损失函数，在计算他的值的时候其实参数已经发生了改变
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad

#数值法求梯度，针对二维的情况
def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):        #当X是矩阵时，idx得到他的行索引，x得到每一行的值
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])#np.nditer的具体用法参考“https://blog.csdn.net/weixin_44690866/article/details/110796170”
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad