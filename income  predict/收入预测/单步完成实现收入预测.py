# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:15:58 2020

@author: 15009
"""


import csv
#from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

with open("1.csv", "rt", encoding="utf-8") as csvfile:
   reader = csv.reader(csvfile)
   rows = [row for row in reader]
   X, y = [], []
   for row in rows:
       for i in range(len(row)) :
           row[i]=float(row[i])
       X.append(row[1:-1])
       y.append(row[-1])
     
num_training = int(0.75 * len(X))
num_test = len(X) - num_training
sc=StandardScaler()
sc.fit(X)
X=sc.transform(X)
# Training data
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])
# Test data
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

w = np.zeros(X_train.shape[1],) #57
b = np.zeros(1,) #1
lamda = 0.001 #正则化惩罚过拟合
max_iter = 1000 #迭代次数
batch_size = 10 #每次选取的样本数
learning_rate = 0.6
num_train = len(y_train)
num_dev = len(y_test)
step =1 #迭代步数
loss_train = [] #训练集损失
loss_validation = [] #测试集损失
train_acc = [] #训练集准确率
test_acc = [] #测试集准确率

def shuffle(X, Y):#打乱X,Y
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize]) #ndarray的参数是数组时，返回一个依参数排序后的数组
 
def sigmoid(z):
     # Use np.clip to avoid overflow\超出的部分就把它强置为边界部分
    s = np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)
    return s
 
def get_prob(X, w, b):
    # the probability to output 1
    return sigmoid(np.add(np.matmul(X, w), b))
 
def loss(y_pred, Y_label, lamda, w):
    #ypred不可能为1，0，所以就不用防止log0出现nan？
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy + lamda * np.sum(np.square(w))
 
def accuracy(Y_pred, Y_label):
    return np.sum(Y_pred == Y_label)/len(Y_pred)

def pri(y):
    plt.figure()
    plt.plot(y)
    plt.show()
    return 0
for epoch in range(max_iter):
    # Random shuffle for each epoch
    X_train, y_train = shuffle(X_train, y_train) #打乱各行数据，这样参数能不易陷入局部最优，模型能够更容易达到收敛。     
 
    # Logistic regression train with batch
    for idx in range(int(np.floor(len(y_train)/batch_size))): #每个batch更新一次
        x_bt = X_train[idx*batch_size:(idx+1)*batch_size] #32*57
        y_bt = y_train[idx*batch_size:(idx+1)*batch_size] #32*1
 
        # Find out the gradient of the loss
        y_bt_pred = get_prob(x_bt, w, b) #matmul：二维数组间的dot
        pred_error = y_bt - y_bt_pred
        w_grad = -np.mean(np.multiply(pred_error, x_bt.T), 1)+lamda*w #multiply：数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
        b_grad = -np.mean(pred_error)
 
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad
        step = step+1
 
    # Compute the loss and the accuracy of the training set and the validation set
    y_pred = get_prob(X_train, w, b)
    yh = np.round(y_pred)
    train_acc.append(accuracy(yh, y_train))
    loss_train.append(loss(y_pred, y_train, lamda, w)/num_train)
    
    y_test_pred = get_prob(X_test, w, b)
    yh_test = np.round(y_test_pred)
    test_acc.append(accuracy(yh_test, y_test))
    loss_validation.append(loss(y_test_pred, y_test, lamda, w)/num_dev)


pri(train_acc)
pri(test_acc)
pri(loss_train)
pri(loss_validation)
