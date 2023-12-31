from matplotlib import pyplot as plt
import numpy as np
loss_single_Relu=np.load('npy_of_basic/loss_single_Relu.npy')
acc_single_Relu=np.load('npy_of_basic/acc_single_Relu.npy')
loss_double_Relu=np.load('npy_of_basic/loss_double_Relu.npy')
acc_double_Relu=np.load('npy_of_basic/acc_double_Relu.npy')
loss_three_Relu=np.load('npy_of_basic/loss_three_Relu.npy')
acc_three_Relu=np.load('npy_of_basic/acc_three_Relu.npy')
loss_zero_Relu=np.load('npy_of_basic/loss_zero_Relu.npy')
acc_zero_Relu=np.load('npy_of_basic/acc_zero_Relu.npy')
loss_single_Sigmoid=np.load('npy_of_basic/loss_single_Sigmoid.npy')
acc_single_Sigmoid=np.load('npy_of_basic/acc_single_Sigmoid.npy')
loss_duble_sigmoid=np.load('npy_of_basic/loss_double_sigmoid.npy')
acc_double_sigmod=np.load('npy_of_basic/acc_double_sigmoid.npy')
plt.figure(1)
plt.plot(acc_single_Relu,label='single',color='red')
plt.plot(acc_double_Relu,label='double',color='blue')
plt.plot(acc_three_Relu,label='three',color='green')
plt.plot(acc_zero_Relu,label='zero',color='black')
plt.legend()
plt.xlabel('iter *600')
plt.ylabel('acc')
plt.title('acc of different layers')
plt.savefig('output_image/acc.png')
plt.figure(2)
plt.plot(loss_single_Relu,label='single+Relu',color='red')
plt.plot(loss_double_Relu,label='double+Relu',color='blue')
plt.plot(loss_three_Relu,label='three+Relu',color='green')
plt.plot(loss_zero_Relu,label='zero+Relu',color='black')
plt.legend()
plt.xlabel('iter *600')
plt.ylabel('loss')
plt.title('loss of different layers')
plt.savefig('output_image/loss.png')
print('acc_single_Relu',acc_single_Relu[-1])
print('acc_double_Relu',acc_double_Relu[-1])
print('acc_three_Relu',acc_three_Relu[-1])
print('acc_zero_Relu',acc_zero_Relu[-1])
print('acc_single_Sigmoid',acc_single_Sigmoid[-1])
print('loss_single_Relu',loss_single_Relu[-1])
print('loss_double_Relu',loss_double_Relu[-1])
print('loss_three_Relu',loss_three_Relu[-1])
print('loss_zero_Relu',loss_zero_Relu[-1])
plt.figure(3)
plt.plot(acc_single_Sigmoid,label='single+Sigmoid',color='red')
plt.plot(acc_double_sigmod,label='double+Sigmoid',color='blue')
plt.plot(acc_single_Relu,label='single+Relu',color='green')
plt.plot(acc_double_Relu,label='double+Relu',color='black')
plt.legend()
plt.xlabel('iter *600')
plt.ylabel('acc')
plt.title('acc of difference activation function')
plt.savefig('output_image/acc_activation.png')
plt.figure(4)
plt.plot(loss_single_Sigmoid,label='single+Sigmoid',color='red')
plt.plot(loss_duble_sigmoid,label='double+Sigmoid',color='blue')
plt.plot(loss_single_Relu,label='single+Relu',color='green')
plt.plot(loss_double_Relu,label='double+Relu',color='black')
plt.legend()
plt.xlabel('iter *600')
plt.ylabel('loss')
plt.title('loss of difference activation function')
plt.savefig('output_image/loss_activation.png')
plt.show()
