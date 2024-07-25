import numpy as np
from optimizer import *
class train:
    def __init__(self,x_train=None,t_train=None,x_test=None,t_test=None,optimizer=SGD(),t1=60000,t2=10000,batch_size=100,max_iter=10000) :
        self.batch_size=batch_size
        self.max_iter=max_iter
        self.x_train=x_train
        self.t_train=t_train
        self.x_test=x_test
        self.t_test=t_test
        self.optimizer=optimizer
        self.t1=t1
        self.t2=t2
        
    def train(self,network,loss_list=None,accuracy_list_test=None,accuracy_list_train=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
        
        train_size=self.x_train.shape[0]
        iter_per_epoch=max(train_size/self.batch_size,1)
        for i in range(self.max_iter):
            batch_mask=np.random.choice(train_size,self.batch_size)
            x_batch=self.x_train[batch_mask]
            t_batch=self.t_train[batch_mask]
            network.update(x_batch,t_batch,self.optimizer)
            loss=None
            test_acc=None
            train_acc=None
            if get_loss:
                loss=network.loss(x_batch,t_batch)
                loss_list.append(loss)
            if i%iter_per_epoch==0 and get_accuracy:
                train_acc=network.accuracy(self.x_train[:self.t1],self.t_train[:self.t1])
                test_acc=network.accuracy(self.x_test[:self.t2],self.t_test[:self.t2])
                accuracy_list_test.append(test_acc)
                if get_accuracy_train:
                    accuracy_list_train.append(train_acc)
                print('train_acc:',train_acc,'test_acc:',test_acc,'loss:',loss)
        print('the final accuracy of test is :',network.accuracy(self.x_test,self.t_test,self.x_test.shape[0]))
        print('the final accuracy of train is :',network.accuracy(self.x_train,self.t_train,self.x_train.shape[0]))