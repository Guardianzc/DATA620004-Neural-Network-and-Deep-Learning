import gzip
import struct
import os 
import numpy as np
def load_mnist_train(path, mode='train'): 
    # 读取文件
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% mode)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% mode)
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    label_matrix = np.zeros((images.shape[0], 10))
    length = len(labels)
    for i in range(length):
        label_matrix[i][labels[i]] = 1
    return images, label_matrix