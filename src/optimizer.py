# coding: utf-8
import numpy as np

class SGD:

    """随机梯度下降法（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, layers):
        for key in layers.keys():
            if layers[key].W is None:
                continue
            layers[key].W -= self.lr * layers[key].dW
            layers[key].b -= self.lr * layers[key].db


class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, layers):
        if self.m is None:
            self.m, self.v = {}, {}
            for name in layers.keys():
                if layers[name].W is None:
                    continue
                self.m[name+'w'] = np.zeros_like(layers[name].W)
                self.m[name+'b'] = np.zeros_like(layers[name].b)
                self.v[name+'w'] = np.zeros_like(layers[name].W)
                self.v[name+'b'] = np.zeros_like(layers[name].b)
            
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        for name in layers.keys():
            if layers[name].W is None:
                continue
            self.m[name+'w'] += (1 - self.beta1) * (layers[name].dW - self.m[name+'w'])
            self.m[name+'b'] += (1 - self.beta1) * (layers[name].db - self.m[name+'b'])
            self.v[name+'w'] += (1 - self.beta2) * (layers[name].dW**2 - self.v[name+'w'])
            self.v[name+'b'] += (1 - self.beta2) * (layers[name].db**2 - self.v[name+'b'])
            layers[name].W -= lr_t * self.m[name+'w'] / (np.sqrt(self.v[name+'w']) + 1e-7)
            layers[name].b -= lr_t * self.m[name+'b'] / (np.sqrt(self.v[name+'b']) + 1e-7)
            
        