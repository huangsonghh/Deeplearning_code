# 开发日期：2023/9/3
#这个文件定义的是数值法求梯度

# 定义一个计算一维梯度的函数
import numpy as np
def gradient1(f, x):  # 一维梯度计算
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):  # 为什么要先去改变x中元素的值然后又去还原它呢？这其实是为之后求参数的偏导做准备
        tmp_val = x[idx]  # 因为改变了值以后，下一步调用f（x），也就是损失函数，在计算他的值的时候其实参数已经发生了改变
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad

# 定义一个计算二维梯度的函数
def gradient2(f, x):
    grad = np.zeros_like(x)
    if np.ndim(x) == 1:
        grad = gradient1(f, x)
    else:
        for i,y in enumerate(x):
            grad[i] = gradient1(f,y)
    return grad

# 可以用于计算多维数组的梯度
def numerical_gradient3(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    #有效的多维迭代器对象，可以遍历数组。
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])  # np.nditer的具体用法参考“https://blog.csdn.net/weixin_44690866/article/details/110796170”
    #'multi_index'是将元素索引（0，0），（1，0）等取出来
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad