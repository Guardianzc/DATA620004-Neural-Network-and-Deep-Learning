from matplotlib import pyplot
import matplotlib.pyplot as plt
import os, sys
import numpy as np
os.chdir(os.path.dirname(sys.argv[0]))

'''
用于作图
'''
with open('Training_Curve.txt', 'r') as f:
    lines = f.readlines()

Accuracy_train_list = list()
loss_train_list = list()
Accuracy_valid_list = list()
loss_valid_list =list()
Accuracy_test_list = list()
loss_test_list = list()
for line in lines:
    Accuracy_train, loss_train, Accuracy_valid, loss_valid, Accuracy_test, loss_test = line.split()
    Accuracy_train_list.append(float(Accuracy_train))
    loss_train_list.append(float(loss_train))
    Accuracy_valid_list.append(float(Accuracy_valid)) 
    loss_valid_list.append(float(loss_valid))
    Accuracy_test_list.append(float(Accuracy_test))
    loss_test_list.append(float(loss_test))

x = range(len(Accuracy_train_list))

plt.plot(x, Accuracy_train_list, ms = 10, label='Accuracy_train')
plt.plot(x, Accuracy_valid_list,  ms = 10, label='Accuracy_valid_list')
plt.plot(x, Accuracy_test_list,  ms = 10, label='Accuracy_test_list')
plt.xlabel('Step') #X轴标签
plt.ylabel("Accuracy") #Y轴标签
plt.legend() 
plt.savefig('.\\fig\\Training_Curve.jpg',dpi = 900)
plt.clf()
x = range(len(Accuracy_train_list))

plt.plot(x, loss_train_list, ms = 10, label='loss_train_list')
plt.plot(x, loss_valid_list,  ms = 10, label='loss_valid_list')
plt.plot(x, loss_test_list,  ms = 10, label='loss_test_list')
plt.xlabel('Step') #X轴标签
plt.ylabel("Loss") #Y轴标签
plt.legend() 
plt.savefig('.\\fig\\Loss.jpg',dpi = 900)
plt.clf()

file_path = './/model_save//'
for f in os.listdir(file_path):
    if 'npy' in f:
        data = np.load(file_path + str(f))
        name = f.split('.')[0]
        data = data.reshape(-1)
        plt.hist(data, bins = 50)
        plt.xlabel('Value') #X轴标签
        plt.ylabel("Frequency") #Y轴标签
        plt.savefig('.\\fig\\' + name + '.jpg',dpi = 900)
        plt.clf()
