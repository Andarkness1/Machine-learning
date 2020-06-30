# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:29:34 2020

@author: 15009
"""
import csv
from sklearn import linear_model
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
       #print(type(row[0]))
num_training = int(0.75 * len(X))
num_test = len(X) - num_training
# Training data
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

# Train the model using the training sets
classifier = linear_model.LogisticRegression(solver='liblinear', C=10000)

# train the classifier
classifier.fit(X_train, y_train)
y_predict= classifier.predict(X_test)
count=0
for i in range(len(y_test)):
    if y_test[i]==y_predict[i] :
        count+=1
    
Accuracy=count*1.0/len(y_test)
print("Accuracy=",Accuracy)
# draw datapoints and boundaries
#plot_classifier(classifier, X, y)

# Predict the output

