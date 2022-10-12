#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import scipy as scp
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply, eigsh, eigs
import scipy.sparse
import itertools as iter
import os
import time

T = 200

Delta0 = 6.
U = 7.
spinno = 5
sparse = False
local = False
if(spinno > 8): sparse = True
    
def component(com, spinno, sparse, local=False):

    pauli_x = np.array([[0,1 + 0j],[1+ 0j,0]])
    pauli_y = np.array([[0,-1j],[1j,0]])
    pauli_z = np.array([[1,0+ 0j],[0,-1]])
    identity = np.array([[1+ 0j,0],[0,1+ 0j]])
    nmat = np.array([[0.,0+ 0j],[0,1]])
        
    if(sparse):
        pauli_x = csr_matrix(pauli_x)
        pauli_y = csr_matrix(pauli_y)
        pauli_z = csr_matrix(pauli_z)
        identity = csr_matrix(identity)
        nmat = csr_matrix(nmat)
    
        if(com == 'zz'):
            op = scp.sparse.kron(pauli_z, pauli_z)
        elif(com == 'x'):
            op = scp.sparse.kron(pauli_x, identity)
            op2 = pauli_x
        elif(com == 'z'):
            op = scp.sparse.kron(pauli_z, identity)
            op2 = pauli_z
        elif(com == 'y'):
            op = scp.sparse.kron(pauli_y, identity)
            op2 = pauli_y
        elif(com == 'n'):
            op = scp.sparse.kron(nmat, identity)
            op2 = nmat
        
    else:
    
        if(com == 'zz'):
            op = np.kron(pauli_z, pauli_z)
        elif(com == 'x'):
            op = np.kron(pauli_x, identity)
            op2 = pauli_x
        elif(com == 'z'):
            op = np.kron(pauli_z, identity)
            op2 = pauli_z
        elif(com == 'y'):
            op = np.kron(pauli_y, identity)
            op2 = pauli_y
        elif(com == 'n'):
            op = np.kron(nmat, identity)
            op2 = nmat
    
    if(sparse):
        
        component = scp.sparse.kron(scp.sparse.kron(scp.sparse.eye(2**0), op), scp.sparse.eye(2**(spinno-2)))
        if(local):
            components = []
            components.append(component)
        else:
            components = 0.
            components += component
        for n in range(2, spinno):
            comp = scp.sparse.kron(scp.sparse.kron(scp.sparse.eye(2**(n-1)), op), scp.sparse.eye(2**(spinno-n-1)))
            if(local): components.append(comp)
            else: components += comp
        if(local and not com == 'zz'): components.append(scp.sparse.kron(scp.sparse.eye(2**(spinno-1)), op2))
        elif(not local and not com == 'zz'): components += scp.sparse.kron(scp.sparse.eye(2**(spinno-1)), op2)
    else:
        component = np.kron(np.kron(np.eye(2**0), op), np.eye(2**(spinno-2)))
        if(local):
            components = np.zeros((spinno, 2**(spinno), 2**(spinno)),dtype=complex)
            components[0] = component
        else:
            components = 0.
            components += component
        for n in range(2, spinno):
            comp = np.kron(np.kron(np.eye(2**(n-1)), op), np.eye(2**(spinno-n-1)))
            if(local): components[n-1] = comp
            else:
                components += comp
        if(local and not com == 'zz'): components[-1] = (np.kron(np.eye(2**(spinno-1)), op2))
        elif(not local and not com == 'zz'): components += np.kron(np.eye(2**(spinno-1)), op2)

    return components
    
def tensor(com, spinno, sparse, local=False):

    hadamard = 1/np.sqrt(2) * np.array([[1,1],[1 + 0j,-1]])
    sdagger = np.array([[1,0.],[0.,0. - 1j]])
    
    if(sparse):
        hadamard = csr_matrix(hadamard)
        sdagger = csr_matrix(sdagger)
        
    if(com == 'had'): op = hadamard
    elif(com == 'sdag'): op = sdagger
    
    if(sparse):
         result = scp.sparse.kron(op, op)
         for n in range(spinno - 2):
            result = scp.sparse.kron(result, op)
    else:
        result = np.kron(op, op)
        for n in range(spinno - 2):
           result = np.kron(result, op)
    
    return result

def Jhnn(spinno, sparse, pairs, Jpairs):

    nmat = np.array([[0.,0+ 0j],[0,1]])
        
    if(sparse):
        nmat = csr_matrix(nmat)
        
    if(sparse): Jhnn = csr_matrix((2**(spinno), 2**(spinno)), dtype=complex)
    else: Jhnn = np.zeros((2**(spinno), 2**(spinno)), dtype=complex)
    
    for i in range(len(Jpairs)):
    
        s1 = pairs[i, 0]
        s2 = pairs[i, 1]
        
        if(sparse): Jhnn += Jpairs[i] * scp.sparse.kron(scp.sparse.kron(scp.sparse.kron(scp.sparse.eye(2**(s1-1)), nmat), scp.sparse.eye(2**(s2-s1-1))), scp.sparse.kron(nmat, scp.sparse.eye(2**(spinno-s2))))
        
        else: Jhnn += Jpairs[i] * np.kron(np.kron(np.kron(np.eye(2**(s1-1)), nmat), np.eye(2**(s2-s1-1))), np.kron(nmat, np.eye(2**(spinno-s2))))
    
    return Jhnn
    
def H(Delta, local, spinno, sparse, pairs, Jpairs):

    hn = component('n', spinno, sparse)
    
    if(sparse): hamiltonian = csr_matrix((2**(spinno), 2**(spinno)), dtype=complex)
    else: hamiltonian = np.zeros((2**(spinno), 2**(spinno)), dtype=complex)
    
    hamiltonian += Jhnn(spinno, sparse, pairs, Jpairs)
    hamiltonian += Delta * hn
            
    return hamiltonian
    
def probabilities(state, spinno, sparse):

    binary = '0'+str(spinno)+'b'
    dim = 2**spinno

    probsx = np.zeros(spinno)
    probsy = np.zeros(spinno)
    probsz = np.zeros(spinno)

    hadamards = tensor('had', spinno, sparse)
    sdags = tensor('sdag', spinno, sparse)

    zmeas = np.abs(state)**2
    xmeas = np.abs(hadamards.dot(state))**2
    ymeas = np.abs(sdags.dot(state))**2
    
    for d in range(dim):
        op = format(d, binary)
        for s in range(spinno):
            if(op[s] == '0'):
                probsx[s] += xmeas[d]
                probsy[s] += ymeas[d]
                probsz[s] += zmeas[d]
                
    return probsx, probsy, probsz
    
for id in range(500):

    print("id: ", id)

    prob = np.random.rand(1)[0]

    pairs = list(iter.combinations(range(1, spinno+1), 2))
    pairs = np.array(pairs)
    edges = np.random.choice(2, len(pairs), p=[prob, 1 - prob])
    Jpairs = U * edges
    numedges = np.sum(edges)
    print("Number of pairs: {}".format(len(pairs)))
    print("Number of edges: {}".format(numedges))
    for i in range(len(edges)):
        if(edges[i] == 1): print(pairs[i])

    Ham = H(-Delta0, local, spinno, sparse, pairs, Jpairs)

    if(sparse):
        eigvals, eigvecs = eigsh(Ham, which='SA', k=100, return_eigenvectors=True)
        gs = eigvecs[:,0]
    else:
        eigvals, eigvecs = la.eigh(Ham)
        gs = eigvecs[:,0]
        n = 1
        while(eigvals[n] == eigvals[0] and n < spinno):
            gs += eigvecs[:,n]
            n += 1
        gs /= np.sqrt(n)
        
        #print(gs)
        
    p0x, p0y, p0z = probabilities(gs, spinno, sparse)

    measurements = np.zeros((spinno, T))

    for s in range(spinno):
        
        # 0 = |+>, 1 = |->, 2 = |+y>, 3 = |-y>, 4 = |0>, 5 = |1>
        measurements[s] = np.random.choice([0,1,2,3,4,5], T, p= np.abs([p0x[s]/3,(1-p0x[s])/3, p0y[s]/3,(1-p0y[s])/3, p0z[s]/3,(1-p0z[s])/3]))

    #measurements = measurements.transpose()
        
    filename = "MIS/mis_{}N_{}T_id{}_shadow.txt".format(spinno, T, id)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, measurements.astype(int), fmt='%i')
    
    filename2 = "MIS/mis_{}N_{}T_id{}_couplings.txt".format(spinno, T, id)
    os.makedirs(os.path.dirname(filename2), exist_ok=True)
    np.savetxt(filename2, edges.astype(int), fmt='%i')
   
    filenamex = "MIS/mis_{}N_{}T_id{}_actualsize.txt".format(spinno, T, id)
    os.makedirs(os.path.dirname(filenamex), exist_ok=True)
    np.savetxt(filenamex, 1-p0z, fmt='%1.12f')
