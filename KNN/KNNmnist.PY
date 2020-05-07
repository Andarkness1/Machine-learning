# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:42:53 2020

@author: 15009
"""


#coding:utf-8

import numpy as np
import os
import gzip
from six.moves import urllib

from datetime import datetime

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#下载mnist数据集，仿照tensorflow的base.py中的写法。
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

#按32位读取，主要为读校验码、图片数量、尺寸准备的
#仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
#仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print (magic, num_images, rows, cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

#该算法是执行newInput与给定的dataset之间的距离，再通过最小距离的k张图，给出占比最大的标签
#本来以为需要新增如果三个相同的改为最近的，后来测试发现不需要加入，因为第一个key就是最小的
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    #np.tile(A,B)：重复A B次，相当于重复[A]*B
    #print np.tile(newInput, (numSamples, 1)).shape
    diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):

        voteLabel = labels[sortedDistIndices[i]]
    #    print(voteLabel)
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
     #   print(classCount)
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
  #  if maxCount == 1:
  #      return labels[sortedDistIndices[0]]
  #  else:
        return maxIndex



maybe_download('train_images', 'data/mnist', SOURCE_URL+TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL+TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL+TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL+TEST_LABELS)



# 主函数，先读图片，然后用于测试手写数字
def testHandWritingClass():
    ## step 1: load data
    print ("step 1: load data...")
    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    ## step 2: testing
    print ("step 2: testing...")
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples/100
    for i in range(int(test_num)):

        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        #把train_x和train_y的数据作为原有的数据点，然后将text_x中数据一个个进行knn比较并得出结果
        if predict == test_y[i]:
            matchCount += 1
        if (i+1) % 100 == 0:
            print ("完成%d张图片"%(i+1))
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print ("一共运行了%d秒"%((b-a).seconds))

    ## step 3: show the result
    print ("step 3: show the result...")
    print ('The classify accuracy is: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    testHandWritingClass()
    
    #部分代码源自CSDN中的一篇文章，进行了一部分修改以及注释
    #原文链接：https://blog.csdn.net/juanjuan1314/java/article/details/77979582
