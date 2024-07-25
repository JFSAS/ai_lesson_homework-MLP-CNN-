import numpy as np
import gzip
import pickle
import os
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"
# 显示mnist数据集的图片
import matplotlib.pyplot as plt
with open(save_file,'rb') as f:
    dataset=pickle.load(f)
    i=0

x_train,x_test=dataset['train_img'],dataset['test_img']
x_label,x_test=dataset['train_label'],dataset['test_label']
##显示图片

for i in  [2, 5, 14, 29, 31, 37, 39, 40, 46, 57, 74, 89, 94, 96, 107, 135]:
    
    img=x_train[i]
    label=x_label[i]
    plt.title(label)
    plt.imshow(img.reshape(28,28),cmap='gray')
    os.makedirs('the_image_of_three_hidden_layer',exist_ok=True)
    plt.savefig('the_image_of_three_hidden_layer/{}.png'.format(i))
