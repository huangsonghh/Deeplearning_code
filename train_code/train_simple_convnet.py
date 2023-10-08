# 开发日期：2023/9/17
import numpy as np
import os, sys
sys.path.append(os.pardir)

from simple_conv_net import SimpleConvNet
from Deeplearning_code.dataset.mnist import load_mnist

#导入数据
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=False)

#超参数初始化
batch_size=100
learning_rate=0.1
weight_step_std=0.01
epoch_num=100
epoch_size=max(1,len(x_train)/batch_size)



network=SimpleConvNet(input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},  # 这里面是卷积层中的超参数
                 hidden_szie=100, output_size=10, weight_int_std=0.01)