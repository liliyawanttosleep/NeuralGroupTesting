# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:51:00 2023

@author: Lenovo
"""
import numpy as np

def hamming_weight(vector):
    """ Calculate the Hamming weight of a vector. """
    return np.sum(vector)

def H(vector):
    """ Return the indices where the vector has non-zero entries. """
    return np.nonzero(vector)[0]

def CalcSelectedRows(A, Q, l):
    """ Calculate the set of rows that intersect with Q. """
    C = set()
    for i in range(l):
        if set(H(A[i])).intersection(Q):
            C.add(i)
    return C

def UpdateWeight(A, Q, C):
    """ Update the weight vector according to the pseudocode. """
    n = A.shape[1]
    if not C:  # If C is empty
        w = np.ones(n, dtype=float)
    else:
        # Sum the rows in A that are indexed by C
        w = np.sum(A[list(C)], axis=0, dtype=float)

    # Set the weights of indices in Q to infinity
    for i in Q:
        w[i] = np.inf

    return w

def SumColumns(A, l, S):
    """ Sum the columns of A. """
    w = np.full(A.shape[1], np.inf)
    for i in S:
        w[i] = np.sum(A[:l, i])
    return w

def RRD(n, m, alpha):
    """ Random Row Design algorithm with float handling. """
    A = np.zeros((m, n), dtype=int)

    # Randomly choose the first row of A
    a = np.random.choice([0, 1], size=n, p=[1 - alpha/n, alpha/n])
    while hamming_weight(a) != alpha:
        a = np.random.choice([0, 1], size=n, p=[1 - alpha/n, alpha/n])
    A[0] = a

    # Construct each row of A
    for l in range(1, m):
        k = 0
        Qk = set()

        # Construct the l-th row
        while k < alpha:
            k += 1
            C = CalcSelectedRows(A, Qk, l)
            w_hat = UpdateWeight(A, Qk, C)
            w_min = np.min(w_hat[w_hat != np.inf])

            S = set(np.where(w_hat == w_min)[0])
            z_hat = SumColumns(A, l, S)
            z_min = np.min(z_hat[z_hat != np.inf])

            X = set(np.where(z_hat == z_min)[0])
            s = np.random.choice(list(X))
            Qk.add(s)
            A[l, s] = 1

    return A

# Example usage
n = 10  # Number of columns
m = 20  # Number of rows
alpha = 8  # Maximum Hamming weight

# Generate the matrix
matrix = RRD(n, m, alpha)
print(matrix)

