#!/usr/bin/env python
# coding: utf-8

# # Predicting ground states for 2D Heisenberg models

# In[1]:


# Basic functionalities
import numpy as np
import random
import copy
import ast
import datetime as dt
from timeit import default_timer as timer
from os import path

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

length = 5 # length = 4, 5, 6, 7, 8, 9 are all available

Xfull = [] # Shape = (number of data) x (number of params)
Ytrain = [] # Shape = (number of data) x (number of pairs), estimated 2-point correlation functions
Yfull = [] # Shape = (number of data) x (number of pairs), exact 2-point correlation functions

for idx in range(1, 101):
    if path.exists('./heisenberg_data/heisenberg_{}x5_id{}_XX.txt'.format(length, idx)) == False:
        continue
    with open('./heisenberg_data/heisenberg_{}x5_id{}_samples.txt'.format(length, idx), 'r') as f:
        single_data = []
        classical_shadow = [[int(c) for i, c in enumerate(line.split("\t"))] for line in f]
        for i in range(length * 5):
            for j in range(length * 5):
                if i == j:
                    single_data.append(1.0)
                    continue
                corr = 0
                cnt = 0
                for shadow in classical_shadow:
                    if shadow[i] // 2 == shadow[j] // 2:
                        corr += 3 if shadow[i] % 2 == shadow[j] % 2 else -3
                    cnt += 1
                single_data.append(corr / cnt)
        Ytrain.append(single_data)
    with open('./heisenberg_data/heisenberg_{}x5_id{}_XX.txt'.format(length, idx), 'r') as f:
        single_data = []
        for line in f:
            for i, c in enumerate(line.split("\t")):
                v = float(c)
                single_data.append(v)
        Yfull.append(single_data)
    with open('./heisenberg_data/heisenberg_{}x5_id{}_couplings.txt'.format(length, idx), 'r') as f:
        single_data = []
        for line in f:
            for i, c in enumerate(line.split("\t")):
                v = float(c)
                single_data.append(v)
        Xfull.append(single_data)

Xfull = np.array(Xfull)
print("number of data (N) * number of params (m) =", Xfull.shape)
Ytrain = np.array(Ytrain)
Yfull = np.array(Yfull)
print("number of data (N) * number of pairs =", Yfull.shape)


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
kernel_NN2 = kernel_fn(Xfull, Xfull, 'ntk')

print(kernel_NN2)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN3 = kernel_fn(Xfull, Xfull, 'ntk')
                
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN4 = kernel_fn(Xfull, Xfull, 'ntk')

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(32), stax.Relu(),
    stax.Dense(1)
)
kernel_NN5 = kernel_fn(Xfull, Xfull, 'ntk')

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

for k in range((length * 5) * (length * 5)):
    # each k corresponds to the correlation function in a pair of qubits
    print("k =", k)

    def train_and_predict(kernel, opt="linear"): # opt="linear" or "rbf"
        
        # instance-wise normalization
        for i in range(len(kernel)):
            kernel[i] /= np.linalg.norm(kernel[i])

        # consider the k-th pair
        global k
        
        # training data (estimated from measurement data)
        y = np.array([Ytrain[i][k] for i in range(len(Xfull))])
        X_train, X_test, y_train, y_test = train_test_split(kernel, y, test_size=0.4, random_state=0)

        # testing data (exact expectation values)
        y_clean = np.array([Yfull[i][k] for i in range(len(Xfull))])
        _, _, _, y_test_clean = train_test_split(kernel, y_clean, test_size=0.4, random_state=0)

        # use cross validation to find the best method + hyper-param
        best_cv_score, test_score = 999.0, 999.0
        for ML_method in [(lambda Cx: svm.SVR(kernel=opt, C=Cx)), (lambda Cx: KernelRidge(kernel=opt, alpha=1/(2*Cx)))]:
            for C in [0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0]:
                score = -np.mean(cross_val_score(ML_method(C), X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"))
                if best_cv_score > score:
                    clf = ML_method(C).fit(X_train, y_train.ravel())
                    test_score = np.linalg.norm(clf.predict(X_test).ravel() - y_test_clean.ravel()) / (len(y_test) ** 0.5)
                    best_cv_score = score
                
        return best_cv_score, test_score

    # Dirichlet
    print("Dirich. kernel", train_and_predict(kernel_dir))
    # RBF
    print("Gaussi. kernel", train_and_predict(Xfull, opt="rbf"))
    # Neural tangent
    for kernel_NN in list_kernel_NN:
        print("Neur. T kernel", train_and_predict(kernel_NN))


# # Classifying topological-ordered and trivial phases

# In[39]:


# Kernel PCA from sklearn
import numpy as np
from sklearn.decomposition import PCA

# Plotting tools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks")


# In[51]:


# Load the shadow kernel matrix (for 100 different states; 50 topological, 50 trivial)
with open('topological_alldep_tr=10.txt', 'r') as f:
    all_K = ast.literal_eval(f.read())

# Perform kernel PCA based on shadow kernel
data = []
for d, K in enumerate(all_K[:-1]):
    X = []
    orig_id = []
    for i in range(len(K)):
        orig_id.append(i)
        single_X = []
        for j in range(len(K)):
            single_X.append(K[i][j] / ((K[i][i] * K[j][j]) ** 0.5))
        X.append(single_X)

    X = np.array(X)
    pca = PCA(n_components=1)
    F = pca.fit_transform(X)
    std = np.std(F)
    
    for z in range(len(X)):
        i = orig_id[z]
        if d == 0 or d == 9:
            data.append((-F[z, 0] / std, d, "Topological" if z < 5 else "Trivial"))
        else:
            data.append((F[z, 0] / std, d, "Topological" if z < 5 else "Trivial"))


# In[52]:


# Plot the 1D representation for different circuit depth
plt.figure(figsize=(3.0, 2.0))
df_toric = pd.DataFrame(data=data, columns = ['PC1', 'Depth', 'Phase'])
ax = sns.stripplot(x="Depth", y="PC1", hue='Phase', data=df_toric[df_toric["Phase"]=="Topological"], palette=['#ED686D'], orient="v",
                   edgecolor="black", marker="o", s=10, alpha=0.85, jitter=0.22)
ax = sns.stripplot(x="Depth", y="PC1", hue='Phase', data=df_toric[df_toric["Phase"]=="Trivial"], palette=['#80A3FA'], orient="v",
                   edgecolor="black", marker="D", s=10, alpha=0.55, jitter=0.22)

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
ax.set_xlabel("Circuit depth")
ax.set_ylabel("1st princ.\n comp.")
ax.set_ylim(-4.0, 4.0);

