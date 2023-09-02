# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:20:14 2023

@author: huawei
"""

import numpy as np

def generate_rrsd_matrix(m, n, r):
    """
    Generate a Random r-Size Design (RrSD) matrix.
    
    Parameters:
    m (int): Number of rows
    n (int): Number of columns
    r (int): Weight of each row
    
    Returns:
    np.array: m x n matrix
    """
    if r > n:
        raise ValueError("The row weight r cannot be greater than the number of columns n.")
    
    # Initialize an empty m x n matrix
    matrix = np.zeros((m, n), dtype=int)
    
    for i in range(m):
        # Randomly select r indices to set to 1
        random_indices = np.random.choice(n, r, replace=False)
        matrix[i, random_indices] = 1
    
    return matrix

# Example usage
m = 5  # Number of rows
n = 10  # Number of columns
r = 3  # Row weight

matrix = generate_rrsd_matrix(m, n, r)
print("Generated RrSD matrix:")
print(matrix)
