# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:29:34 2020

@author: 15009
"""
import csv
from sklearn import linear_model

import matplotlib.pyplot as plt
import numpy as np

'''def plot_classifier(classifier, X, y):
    # define ranges to plot the figure 
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()

    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()'''

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

