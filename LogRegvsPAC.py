# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:56:28 2019

@author: Hans
"""
%reset -f
#%% create a random dataset

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Create the dataset, all features are composed out of data drawn from gaussian distributions
X, Y = make_classification(n_samples=5000, 
                           n_features=4, 
                           n_informative=2, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2, 
                           n_clusters_per_class=2)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1000)

#%% We perform logistic regression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
print('Logistic Regression score: {}'.format(lr.score(X_test, Y_test)))

#%% Passive Aggressive Classification

import matplotlib.pyplot as plt


Y_train[Y_train==0] = -1
Y_test[Y_test==0] = -1

C = 0.01 # softens the classification
w = np.zeros((X_train.shape[1], 1))

# Implement a Passive Aggressive Classification
for i in range(X_train.shape[0]):
    xi = X_train[i].reshape((X_train.shape[1], 1))
    
    loss = max(0, 1 - (Y_train[i] * np.dot(w.T, xi))) # implement the Hinge Loss
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C))) #2-norm (largest sing. value)
    
    coeff = tau * Y_train[i]
    w += coeff * xi
    
# Compute accuracy
Y_pred = np.sign(np.dot(w.T, X_test.T))
c = np.count_nonzero(Y_pred - Y_test)

print('PA accuracy: {}'.format(1 - float(c) / X_test.shape[0]))



# after one run we obtain the same result

#%% Regression
"""
Instead of a False and True outcome, we have a continious outcome; such as a percentage
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression

# Create the dataset
X, Y = make_regression(n_samples=5000, 
                       n_features=4)

# Implement a Passive Aggressive Regression
C = 0.01 #agressiveness of the correction
eps = 0.1 # introduce a tollerance for small errors
w = np.zeros((X.shape[1], 1))
errors = []

for i in range(X.shape[0]):
    xi = X[i].reshape((X.shape[1], 1))
    yi = np.dot(w.T, xi)
    
    loss = max(0, np.abs(yi - Y[i]) - eps)
    
    tau = loss / (np.power(np.linalg.norm(xi, ord=2), 2) + (1 / (2*C)))
    
    coeff = tau * np.sign(Y[i] - yi) # since continious outcome, and no longer -1 or 1
    errors.append(np.abs(Y[i] - yi)[0, 0])
    
    w += coeff * xi
    
# Show the error plot
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(errors)
ax.set_xlabel('Time')
ax.set_ylabel('Error')
ax.set_title('Passive Aggressive Regression Absolute Error')
ax.grid()

plt.show()

# play with C and eps to optimise results