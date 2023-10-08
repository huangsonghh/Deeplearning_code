# 开发日期：2023/9/5
import numpy as np
import os,sys
sys.path.append(os.pardir)

from com_function import *
from Deeplearning_code.common.util import im2col, col2im

#乘法层，针对两个数值的乘法
class MulLayer:
    def __init__(self):
        self.a=None
        self.b=None

    def forward(self,x,y):
        self.a=x
        self.b=y
        return x*y

    def backward(self,dout):
        dx=dout*self.b
        dy=dout*self.a
        return dx,dy

#加法层
class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        return x+y

    def backward(self,dout):
        return dout,dout

#Relu函数层
class Relu:
    def __init__(self):
        self.mask=None

    def forward(self,x):    #对于一维或者二维数组都适用
        self.mask=(x<=0)    #把 x中小于等于0的部分记录下来
        out=x.copy()        #注意这里不能直接用x[self.mask]=0,由于数组的特性他会发生改变
        out[self.mask]=0
        return out

    def backward(self,dout):
        dout[self.mask]=0         # x<=0的时候是0，其余是dout*1
        dx=dout
        return dx

#sigmoid函数层
class Sigmoid:
    def __init__(self):
        self.out=None

    def forward(self,x):
        self.out=1/(1+np.exp(-x))
        return self.out

    def backward(self,dout):
        dx=self.out*(1-self.out)*dout
        return dx

#批（batch）版的Affine层
#注意这里的Affine层和之前有些不同，它的输入可能是（三维或四维）张量也可能是（一维或二维）矩阵，但输出是矩阵。
class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b
        self.x=None
        self.x_shape=None
        self.dW=None   #这里定义self.dw和self.db是为了后面输出时只用输出dx，其余两个可以通过实例访问
        self.db=None

    def forward(self,x):#当x是张量时我们得先将x展开成一个矩阵行的数量等于数据数量，列的数量等于一张图片的size
        self.x_shape=x.shape
        self.x=x
        y=x.reshape(x.shape[0],-1)   #x.shape=(N,C,H,W)
        out=np.dot(y,self.W)+self.b
        return out

    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        dx=dx.reshape(*self.x_shape)#注意这个地方你不能写成dx=dx.reshape(self.x_shape)，因为括号里只能填数字参数了，不能再填一个元组了，self.x_shape是一个元组
        return dx

#softmax层和loss层
class SoftmaxWithLoss:
    def __init__(self):
        self.y=None
        self.t=None
        self.loss=None   #后续只需要从SoftmaxWithLoss的实例中调用loss函数值就可以了

    def forward(self,x,t):
        self.y=softmax(x)
        self.t=t
        self.loss=cross_entropy_error(self.y,self.t)#交叉熵损失函数
        return self.loss

    def backward(self,dout=1):
        batch_size=self.y.shape[0]
        if np.ndim(self.t)!=1:
            dx = dout * (self.y - self.t) / batch_size
        else:
            dx=self.y.copy()
            dx[np.arange(len(t)),t]-=1
            dx=dx/batch_size
        return dx

#卷积层
class Convolution:
    def __init__(self,W,b,stride,pad):  #这个地方的步幅和填充可以更改
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad

        #backward中要用到的中间量
        self.x=None
        self.col=None
        self.col_W=None

        #权重和偏置的梯度
        self.dW=None
        self.db=None

    def forward(self,x):
        FN,C,FH,FW=self.W.shape
        N,C,H,W=x.shape

        out_h=int((H-FH+2*self.pad)/self.stride+1)   #注意这里要将其整数化
        out_w=int((W-FW+2*self.pad)/self.stride+1)

        col=im2col(x,FH,FW,self.stride,self.pad)#im2col得到的矩阵的大小是(N*out_h*out_w,C*FH*FW)
        col_W=self.W.reshape(FN,-1).T #因为有FN个滤波器，如果是一个滤波器的话这里只需要乘一个列向量。还要注意这里要转置
        self.x=x
        self.col=col
        self.col_W=col_W

        out=np.dot(col,col_W)+self.b
        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        return out

    def backward(self,dout):
        FN,C,FH,FW=self.W.shape
        dout=dout.transpose(0,2,3,1).reshape(-1,FN)

        self.db=np.sum(dout,axis=0)
        self.dW=np.dot(self.col.T,dout)
        self.dW=self.dW.transpose(1,0).reshape(FN,C,FH,FW)

        dcol=np.dot(dout,self.col_W.T)
        dx=col2im(dcol,self.x.shape,FH,FW,self.stride,self.pad)
        return dx


#池化层的实现
class Pooling:
    def __init__(self,pool_h,pool_w,stride,pad):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad

        self.arg_max=None
        self.x=None

    def forward(self,x):
        N,C,H,W=x.shape

        out_h=int((H-self.pool_h+2*self.pad)/self.stride+1)
        out_w=int((W-self.pool_w+2*self.pad)/self.stride+1)

        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)#im2col得到的矩阵的大小是(N*out_h*out_w,C*pool_h*pool_w)
        col=col.reshape(-1,self.pool_h*self.pool_w)#这里得到的大小为(N*out_h*out_w*C , pool_h*pool_w)

        arg_max=np.argmax(col,axis=1)
        out=np.max(col,axis=1)
        out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        self.x=x
        self.arg_max=arg_max

        return out

    def backward(self,dout):
        N,C,H,W=dout.shape
        dout=dout.transpose(0,2,3,1)
        pool_size=self.pool_h*self.pool_w

        dmax=np.zeros((dout.size,pool_size))#注意使用np.zeros时应该输入的是一个形状参数
        dmax[np.arange(N*H*W*C),self.arg_max.flatten()]=dout.flatten()#因为arg_max是一个二维矩阵，所以这里要展开一下
        dmax=dmax.reshape(dout.shape+(pool_size,))#两个元组相加相当于拼接，这里是把dmax从矩阵变为了5维张量,大小是(N，H，W，C,pool_h*pool_w)

        dcol=dmax.reshape(N*H*W,-1)
        dx=col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.stride,self.pad)

        return dx









#下面是一个apple和oringe的例子
# apple_num=2
# apple=100
# oringe_num=3
# oringe=150
# tax=1.1
#
# apple_price=MulLayer()
# oringe_price=MulLayer()
# apple_and_oringe_price=AddLayer()
# sum_tax=MulLayer()
#
# #forward
# out1=apple_price.forward(apple_num,apple)
# out2=oringe_price.forward(oringe_num,oringe)
# out3=apple_and_oringe_price.forward(out1,out2)
# out4=sum_tax.forward(out3,tax)
#
# #backward
# dprice=1
# dout3,dtax=sum_tax.backward(dprice)
# dout1,dout2=apple_and_oringe_price.backward(dout3)
# dapple,dapple_num=apple_price.backward(dout1)
# doringe,doringe_num=oringe_price.backward(dout2)
#
# print(dapple,doringe)
