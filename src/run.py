import numpy as np
from Network import Network
from load_minist import load_mnist
from optimizer import SGD,Adam
import layers
from matplotlib import pyplot as plt
from train import train
import os
def returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train):
    if get_accuracy :
        if get_loss :
            if get_accuracy_train :
                return loss_list,acc_list_test,acc_list_train
            else : 
                return loss_list,acc_list_test
        else :
            if get_accuracy_train :
                return acc_list_test,acc_list_train
            else : 
                return acc_list_test
def CNN(loss_list,acc_list):
    network=Network()
    input_size=28
    filter_size=5
    filter_num=30
    filter_pad=0
    filter_stride=1
    network.add('Conv1',layers.Convolution(filter_num,1,filter_size,filter_size,filter_stride,filter_pad))
    network.add('Relu1',layers.Relu())
    network.add('Pooling1',layers.Pooling(2,2,2))
    #28-5+1
    #12*12*30=4320
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
    network.add('Affien1',layers.Affine(pool_output_size,128))
    network.add('Relu2',layers.Relu())
    network.add('Affien2',layers.Affine(128,10))
    # 权重可能过小
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    x_train=x_train.reshape(-1,1,28,28)
    x_test=x_test.reshape(-1,1,28,28)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam(),t1=1000,t2=1000)
    trainer.train(network,loss_list,acc_list)
    return loss_list,acc_list


def MLP_zero_layer_Relu(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add('Affien1',layers.Affine(784,10))
    # 权重可能过小
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam())
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)

def MLP_one_layer_Relu(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add('Affien1',layers.Affine(784,1000))
    network1.add('Relu1',layers.Relu())
    network1.add('Affien2',layers.Affine(1000,10))
    # 权重可能过小
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam(),max_iter=30000)
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)

def MLP_two_layer_Relu(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add('Affien1',layers.Affine(784,2000))
    network1.add('Relu1',layers.Relu())
    network1.add('Affien2',layers.Affine(2000,2000))
    network1.add('Relu2',layers.Relu())
    network1.add('Affien3',layers.Affine(2000,10))
    # 权重可能过小
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam())
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)

def MLP_three_layer_Relu(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add('Affien1',layers.Affine(784,256))
    network1.add('Relu1',layers.Relu())
    network1.add('Affien2',layers.Affine(256,128))
    network1.add('Relu2',layers.Relu())
    network1.add('Affien3',layers.Affine(128,50))
    network1.add('Relu3',layers.Relu())
    network1.add('Affien4',layers.Affine(50,10))
    # 权重可能过小
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam())
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)


def MLP_one_layer_ReLu_dropout(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add('Affien1',layers.Affine(784,1000))
    network1.add('Relu1',layers.Relu())
    network1.add('Dropout1',layers.Dropout(0.2))
    network1.add('Affien2',layers.Affine(1000,10))
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam(),max_iter=30000)
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)


def MLP_two_layer_ReLu_dropout(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False):
    network1=Network()
    network1.add("Dropout0",layers.Dropout(0.2))
    network1.add('Affien1',layers.Affine(784,2000))
    network1.add('Relu1',layers.Relu())
    network1.add('Dropout1',layers.Dropout(0.5))
    network1.add('Affien2',layers.Affine(2000,2000))
    network1.add('Relu2',layers.Relu())
    network1.add('Dropout2',layers.Dropout(0.5))
    network1.add('Affien3',layers.Affine(2000,10))
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam())
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)


def MLP_three_layer_ReLu_dropout(loss_list=None,acc_list_train=None,acc_list_test=None,get_accuracy=True,get_loss=True,get_accuracy_train=False): 
    network1=Network()
    network1.add('Affien1',layers.Affine(784,256))
    network1.add('Relu1',layers.Relu())
    network1.add('Dropout1',layers.Dropout(0.5))
    network1.add('Affien2',layers.Affine(256,128))
    network1.add('Relu2',layers.Relu())
    network1.add('Dropout2',layers.Dropout(0.5))
    network1.add('Affien3',layers.Affine(128,50))
    network1.add('Relu3',layers.Relu())
    network1.add('Dropout3',layers.Dropout(0.5))
    network1.add('Affien4',layers.Affine(50,10))
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=True)
    trainer=train(x_train,t_train,x_test,t_test,optimizer=Adam())
    trainer.train(network1,loss_list,acc_list_test,acc_list_train,get_accuracy=get_accuracy,get_loss=get_loss,get_accuracy_train=get_accuracy_train)
    return returnlist(get_accuracy,get_loss,get_accuracy_train,loss_list,acc_list_test,acc_list_train)

if __name__ == '__main__' :
    loss_list=[]
    acc_list_test=[]
    acc_list_train=[]
    acc_list_test_dropout=[]
    acc_list_train_dropout=[]

    #MLP(loss_list,acc_list)
    #MLP_zero_layer_Relu(loss_list,acc_list)
    #MLP_one_layer_Relu(loss_list,acc_list)
    #MLP_two_layer_Relu(loss_list,acc_list)
    #MLP_three_layer_Relu(loss_list,acc_list)
    ##探究dropout的效果，需要得到acc_train和acc_test进行对比，画在同一个图上
    print('train one layer with dropout')
    MLP_two_layer_ReLu_dropout(None,acc_list_train_dropout,acc_list_test_dropout,get_accuracy=True,get_loss=False,get_accuracy_train=True)
    print('train one layer without dropout')
    MLP_two_layer_Relu(None,acc_list_train,acc_list_test,get_accuracy=True,get_loss=False,get_accuracy_train=True)
    acc_list_test_dropout = [(1 - acc) * 60000 for acc in acc_list_test_dropout]
    acc_list_train_dropout = [(1 - acc) * 60000 for acc in acc_list_train_dropout]
    acc_list_test = [(1 - acc) * 60000 for acc in acc_list_test]
    acc_list_train = [(1 - acc) * 60000 for acc in acc_list_train]
    plt.figure(1)
    plt.plot(acc_list_test_dropout, label='test_error+dropout', color='red')
    plt.plot(acc_list_train_dropout, label='train_error+dropout', color='blue')
    plt.plot(acc_list_test, label='test_error', color='green')
    plt.plot(acc_list_train, label='train_error', color='yellow')

    plt.legend()
    plt.title('the effect of dropout')
    plt.xlabel('epoch')
    plt.ylabel('error num')
    os.makedirs('npy_dropout', exist_ok=True)
    np.save('npy_dropout/acc_list_test_dropout.npy', acc_list_test_dropout)
    np.save('npy_dropout/acc_list_train_dropout.npy', acc_list_train_dropout)
    np.save('npy_dropout/acc_list_test.npy', acc_list_test)
    np.save('npy_dropout/acc_list_train.npy', acc_list_train)
    plt.show()


    '''
    np.save('npy_of_dropout/loss_one_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_one_layer_ReLu_dropout.npy',acc_list)
    loss_list=[]
    acc_list=[]
    MLP_two_layer_ReLu_dropout(loss_list,acc_list)
    np.save('npy_of_dropout/loss_two_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_two_layer_ReLu_dropout.npy',acc_list)
    loss_list=[]
    acc_list=[]
    MLP_three_layer_ReLu_dropout(loss_list,acc_list)
    np.save('npy_of_dropout/loss_three_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_three_layer_ReLu_dropout.npy',acc_list)
    '''
    plt.show()


    '''
    np.save('npy_of_dropout/loss_one_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_one_layer_ReLu_dropout.npy',acc_list)
    loss_list=[]
    acc_list=[]
    MLP_two_layer_ReLu_dropout(loss_list,acc_list)
    np.save('npy_of_dropout/loss_two_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_two_layer_ReLu_dropout.npy',acc_list)
    loss_list=[]
    acc_list=[]
    MLP_three_layer_ReLu_dropout(loss_list,acc_list)
    np.save('npy_of_dropout/loss_three_layer_ReLu_dropout.npy',loss_list)
    np.save('npy_of_dropout/acc_three_layer_ReLu_dropout.npy',acc_list)
'''