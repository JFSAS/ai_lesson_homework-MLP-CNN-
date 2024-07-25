#打印使用dropout后的准确率
import matplotlib.pyplot as plt
import numpy as np
import os
acc_one_layer_ReLu_dropout=np.load('npy_of_dropout/acc_one_layer_ReLu_dropout.npy')
loss_one_layer_ReLu_dropout=np.load('npy_of_dropout/loss_one_layer_ReLu_dropout.npy')
acc_two_layer_ReLu_dropout=np.load('npy_of_dropout/acc_two_layer_ReLu_dropout.npy')
loss_two_layer_ReLu_dropout=np.load('npy_of_dropout/loss_two_layer_ReLu_dropout.npy')
acc_three_layer_ReLu_dropout=np.load('npy_of_dropout/acc_three_layer_ReLu_dropout.npy')
loss_three_layer_ReLu_dropout=np.load('npy_of_dropout/loss_three_layer_ReLu_dropout.npy')
os.makedirs('output_image_dropout',exist_ok=True)

os.makedirs('output_image_dropout',exist_ok=True)
plt.figure(1)
plt.plot(acc_one_layer_ReLu_dropout*60000,label='one',color='red')
plt.plot(acc_two_layer_ReLu_dropout*60000,label='two',color='blue')
plt.plot(acc_three_layer_ReLu_dropout*60000,label='three',color='green')
plt.legend()
plt.title('acc num of different layers')
plt.xlabel('iter *600')
plt.ylabel('acc num')
plt.savefig('output_image_dropout/acc_num.png')
plt.figure(2)
plt.plot(loss_one_layer_ReLu_dropout,label='one',color='red')
plt.plot(loss_two_layer_ReLu_dropout,label='two',color='blue')
plt.plot(loss_three_layer_ReLu_dropout,label='three',color='green')
plt.legend()
plt.title('loss of different layers')
plt.xlabel('iter *600')
plt.ylabel('loss')
plt.savefig('output_image_dropout/loss.png')
print('acc_one_layer_ReLu_dropout',acc_one_layer_ReLu_dropout[-1])
print('acc_two_layer_ReLu_dropout',acc_two_layer_ReLu_dropout[-1])
print('acc_three_layer_ReLu_dropout',acc_three_layer_ReLu_dropout[-1])