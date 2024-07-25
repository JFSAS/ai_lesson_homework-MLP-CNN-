# coding: utf-8
import numpy as np
from layers import *
from collections import OrderedDict


class Network:

    def __init__(self):
        # 生成层字典
        self.layers = OrderedDict()
        #输出激活层
        self.lastLayer = SoftmaxWithLoss()
    # 向前传播    
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # 增加层
    def add(self, layer_name, layer) :
        self.layers[layer_name] = layer
    # 计算损失函数
    def loss(self, x, t):
        y = self.forward(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t,batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accuracy = 0.0
        for i in range(0,int(x.shape[0]/batch_size)):
            y = self.forward(x[batch_size*i:batch_size*(i+1)])
            t_batch=t[batch_size*i:batch_size*(i+1)]
            y = np.argmax(y, axis=1) 
            accuracy += np.sum(y == t_batch)
        return accuracy/x.shape[0]
    def accuracy_num(self,x,t):
        idxOfimage=[]
        y= self.forward(x)
        y=np.argmax(y,axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        for i,j in enumerate(y):
            if j==t[i]:
                idxOfimage.append(i)
        return idxOfimage
    def update(self, x, t, optimizer):
        # 向后传播
        self.backword(x, t)
        # 更新参数
        optimizer.update(self.layers)   
    def backword(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.keys())
        layers.reverse()
        for layer_name in layers:
            #更新dw,db
            dout = self.layers[layer_name].backward(dout)
