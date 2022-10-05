#!/usr/bin/env python
# coding: utf-8

# # Predicting MIS size using Rydberg Hamiltonians

# Basic functionalities
import numpy as np
import random
import copy
import ast
import datetime as dt
from timeit import default_timer as timer
from os import path
import itertools as iter

# Neural tangent kernel
import jax
from neural_tangents import stax

# Traditional ML methods and techniques
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor


# In[2]:


# Load data

spinno = 5
T = 100

Xfull = [] # Shape = (number of data) x (number of params)
Ytrain = [] # Shape = (number of data) x (number of pairs), estimated 2-point correlation functions
Yfull = [] # Shape = (number of data) x (number of pairs), exact 2-point correlation functions

for idx in range(500):
    if path.exists('./MIS/mis_{}N_{}T_id{}_shadow.txt'.format(spinno, T, idx)) == False:
        continue
    with open('./MIS/mis_{}N_{}T_id{}_shadow.txt'.format(spinno, T, idx), 'r') as f:
        single_data = []
        classical_shadow = [[int(c) for i, c in enumerate(line.split(" "))] for line in f]
        for shadow in classical_shadow:
            estimator = 0.
            for i in range(T):
                if(shadow[i] == 4): estimator -= 1.
                elif(shadow[i] == 5): estimator += 2.
                else: estimator += 1/2
                #print("id: ", idx)
                #print(estimator)
            single_data.append(estimator/T)
        Ytrain.append(single_data)
    with open("MIS/mis_{}N_{}T_id{}_actualsize.txt".format(spinno, T, idx), 'r') as f:
        single_data = []
        for line in f:
            for i, c in enumerate(line.split(" ")):
                v = float(c)
                single_data.append(v)
        Yfull.append(single_data)
    with open("MIS/mis_{}N_{}T_id{}_couplings.txt".format(spinno, T, idx), 'r') as f:
        single_data = []
        for line in f:
            for i, c in enumerate(line.split(" ")):
                v = float(c)
                single_data.append(v)
        Xfull.append(single_data)

Xfull = np.array(Xfull)
print("number of data (N) * number of params (m) =", Xfull.shape)
Ytrain = np.array(Ytrain)
Yfull = np.array(Yfull)
print("number of data (N) * number of pairs =", Yfull.shape)

#print(Xfull)
#print(Ytrain)
#print(Yfull)


# In[3]:


#
# Dirichlet kernel
#

kernel_dir = np.zeros((len(Xfull), Xfull.shape[1]*5))
for i, x1 in enumerate(Xfull):
    cnt = 0
    for k in range(len(x1)):
        for k1 in range(-2, 3):
            kernel_dir[i, cnt] += np.cos(np.pi * k1 * x1[k])
            cnt += 1
print("constructed Dirichlet kernel")

#
# Neural tangent kernel
#
    
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN2 = np.array(kernel_fn(Xfull, Xfull, 'ntk')) + 1e-13

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN3 = np.array(kernel_fn(Xfull, Xfull, 'ntk')) + 1e-13
#print("kernel3: ", kernel_NN3)
                
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN4 = np.array(kernel_fn(Xfull, Xfull, 'ntk')) + 1e-13

#print("kernel4: ", kernel_NN4)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN5 = np.array(kernel_fn(Xfull, Xfull, 'ntk')) + 1e-13

#print("kernel5: ", kernel_NN5)

list_kernel_NN = [kernel_NN2, kernel_NN3, kernel_NN4, kernel_NN5]


for r in range(len(list_kernel_NN)):
    for i in range(len(list_kernel_NN[r])):
        for j in range(len(list_kernel_NN[r])):
            list_kernel_NN[r][i][j] /= (list_kernel_NN[r][i][i] * list_kernel_NN[r][j][j]) ** 0.5

print("constructed neural tangent kernel")
            
#
# RBF kernel is defined in Sklearn
#
print("RBF kernel (will be constructed in sklearn)")


# In[2]:

train_idx, test_idx, _, _ = train_test_split(range(len(Xfull)), range(len(Xfull)), test_size=0.4, random_state=0)

models = []
kernels = []

for k in range(spinno):
    # each k corresponds to the correlation function in a pair of qubits
    print("k =", k)

    def train_and_predict(kernel, opt="linear"): # opt="linear" or "rbf"
        
        # instance-wise normalization
        for i in range(len(kernel)):
            #print(kernel[i])
            kernel[i] /= np.linalg.norm(kernel[i])
        
        # consider the k-th pair
        global k
        
        # training data (estimated from measurement data)
        y = np.array([Ytrain[i][k] for i in range(len(Xfull))])
        X_train, X_test, y_train, y_test = train_test_split(kernel, y, test_size=0.4, random_state=0)
        
        #print("X test shape:", X_test.shape)
        #print("Y test shape:", y_test.shape)

        # testing data (exact expectation values)
        y_clean = np.array([Yfull[i][k] for i in range(len(Xfull))])
        _, _, _, y_test_clean = train_test_split(kernel, y_clean, test_size=0.4, random_state=0)

        # use cross validation to find the best method + hyper-param
        best_cv_score, test_score = 999.0, 999.0
        best_model = KernelRidge(kernel=opt, alpha=1/(2))
        for ML_method in [(lambda Cx: svm.SVR(kernel=opt, C=Cx)), (lambda Cx: KernelRidge(kernel=opt, alpha=1/(2*Cx)))]:
            for C in [0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
                score = -np.mean(cross_val_score(ML_method(C), X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"))
                #print("score: ", score)
                if best_cv_score > score:
                    clf = ML_method(C).fit(X_train, y_train.ravel())
                    prediction = clf.predict(X_test).ravel()
                    test_score = np.linalg.norm(prediction - y_test_clean.ravel()) / (len(y_test) ** 0.5)
                    best_model = clf
                    best_cv_score = score
        
        #for i in range(len(prediction)):
        #    print("Original value: ", y_test_clean.ravel()[i])
        #    print("Prediction: ", prediction.ravel()[i])
        #    print("------")
        
        return best_cv_score, test_score, best_model

    # Dirichlet
    b = 900
    a1, b1, best_model = train_and_predict(kernel_dir)
    print("Dirich. kernel", a1, b1)
    if(b1 < b) : b = b1
    # RBF
    #print("Gaussi. kernel", train_and_predict(Xfull, opt="rbf"))
    # Neural tangent
    for kernel_NN in list_kernel_NN:
        a1, b1, model = train_and_predict(kernel_NN)
        if(b1 < b):
            b = b1
            best_model = model
        print("Neur. T kernel", a1, b1)
        
    models.append(best_model)
    kernels.append()
    
for id in range(10):

    prob = np.random.randint(500, size=1)

    pairs = list(iter.combinations(range(1, spinno+1), 2))
    pairs = np.array(pairs)
    edges = kernel[prob[0]]
    for i in range(len(edges)):
        if(edges[i] == 1): print(pairs[i])
        
    predicted_mis = 0.
    for i in range(spinno):
        prediction = models[i].predict(edges)
        predicted_mis += prediction
        
    print("Predicted MIS size: ", predicted_mis)
    print("-----")
